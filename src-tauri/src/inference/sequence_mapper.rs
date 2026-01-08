//! Sequence mapping via belief propagation on assignment graph.
//!
//! This module runs inference to determine the optimal mapping of
//! spin systems to residue positions in the sequence.

use std::collections::HashMap;

use ndarray::Array1;
use petgraph::Direction;
use uuid::Uuid;

use super::assignment_graph::{AssignmentFactorGraph, AssignmentFactorType, AssignmentNode};
use crate::data::SpinSystem;

/// Result of sequence mapping for one spin system.
#[derive(Debug, Clone)]
pub struct MappingResult {
    pub spin_system_id: Uuid,
    pub assigned_residue: i32,
    pub confidence: f64,
    pub alternatives: Vec<(i32, f64)>, // (residue_seq_code, probability)
}

/// Parameters for sequence mapping.
#[derive(Debug, Clone)]
pub struct MappingParams {
    pub max_iterations: usize,
    pub convergence_threshold: f64,
    pub damping: f64,
}

impl Default for MappingParams {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-6,
            damping: 0.5,
        }
    }
}

/// Run belief propagation on the assignment graph and extract mappings.
pub fn map_to_sequence(
    graph: &mut AssignmentFactorGraph,
    params: &MappingParams,
) -> Vec<MappingResult> {
    // Initialize messages
    initialize_messages(graph);

    // Run belief propagation
    run_assignment_bp(graph, params);

    // Extract marginals and assignments
    extract_assignments(graph)
}

/// Initialize all messages to uniform.
fn initialize_messages(graph: &mut AssignmentFactorGraph) {
    let domain_size = graph.domain_size();
    let uniform = Array1::from_elem(domain_size, 1.0 / domain_size as f64);

    // Collect edges first to avoid borrow conflict
    let edges: Vec<_> = graph
        .graph()
        .edge_indices()
        .map(|e| graph.graph().edge_endpoints(e).unwrap())
        .collect();

    for (from, to) in edges {
        graph.messages_mut().insert((from, to), uniform.clone());
    }
}

/// Run belief propagation on the assignment graph.
fn run_assignment_bp(graph: &mut AssignmentFactorGraph, params: &MappingParams) {
    let domain_size = graph.domain_size();

    for iter in 0..params.max_iterations {
        let mut max_change = 0.0f64;

        // Collect all edges
        let edges: Vec<_> = graph.graph().edge_indices().collect();

        for edge in edges {
            let (from, to) = graph.graph().edge_endpoints(edge).unwrap();

            let new_message = compute_message(graph, from, to, domain_size);

            // Apply damping
            let old_message = graph.messages().get(&(from, to)).cloned();
            let damped_message = if let Some(old) = old_message {
                &old * params.damping + &new_message * (1.0 - params.damping)
            } else {
                new_message.clone()
            };

            // Normalize
            let sum: f64 = damped_message.iter().sum();
            let normalized = if sum > 0.0 {
                &damped_message / sum
            } else {
                Array1::from_elem(domain_size, 1.0 / domain_size as f64)
            };

            // Track convergence
            if let Some(old) = graph.messages().get(&(from, to)) {
                let change: f64 = (&normalized - old).mapv(|x| x.abs()).sum();
                max_change = max_change.max(change);
            }

            graph.messages_mut().insert((from, to), normalized);
        }

        if max_change < params.convergence_threshold {
            tracing::info!(
                "Assignment BP converged after {} iterations (max_change={:.2e})",
                iter + 1,
                max_change
            );
            return;
        }
    }

    tracing::warn!(
        "Assignment BP did not converge after {} iterations",
        params.max_iterations
    );
}

/// Compute a single message from node `from` to node `to`.
fn compute_message(
    graph: &AssignmentFactorGraph,
    from: petgraph::graph::NodeIndex,
    to: petgraph::graph::NodeIndex,
    domain_size: usize,
) -> Array1<f64> {
    let g = graph.graph();
    let messages = graph.messages();

    match &g[from] {
        AssignmentNode::SpinSystemVariable { .. } => {
            // Variable to factor: product of incoming messages except from target
            let mut product = Array1::from_elem(domain_size, 1.0);

            for neighbor in g.neighbors_directed(from, Direction::Incoming) {
                if neighbor != to {
                    if let Some(msg) = messages.get(&(neighbor, from)) {
                        product = &product * msg;
                    }
                }
            }

            product
        }
        AssignmentNode::AssignmentFactor {
            factor_type,
            connected_vars,
        } => {
            // Factor to variable: marginalize over other variables
            compute_factor_message(factor_type, connected_vars, to, graph, domain_size)
        }
    }
}

/// Compute message from factor to variable.
fn compute_factor_message(
    factor_type: &AssignmentFactorType,
    connected_vars: &[petgraph::graph::NodeIndex],
    target_var: petgraph::graph::NodeIndex,
    graph: &AssignmentFactorGraph,
    domain_size: usize,
) -> Array1<f64> {
    match factor_type {
        AssignmentFactorType::TypingPrior { scores } => {
            // Single variable factor - just return the scores
            Array1::from_vec(scores.clone())
        }

        AssignmentFactorType::SequentialConnectivity { confidence } => {
            // Binary factor: if var1=i, var2 should be i+1
            compute_sequential_message(connected_vars, target_var, *confidence, graph, domain_size)
        }

        AssignmentFactorType::Uniqueness => {
            // Binary factor: var1 != var2
            compute_uniqueness_message(connected_vars, target_var, graph, domain_size)
        }
    }
}

/// Compute message for sequential connectivity factor.
fn compute_sequential_message(
    connected_vars: &[petgraph::graph::NodeIndex],
    target_var: petgraph::graph::NodeIndex,
    confidence: f64,
    graph: &AssignmentFactorGraph,
    domain_size: usize,
) -> Array1<f64> {
    if connected_vars.len() != 2 {
        return Array1::from_elem(domain_size, 1.0);
    }

    let (from_var, to_var) = (connected_vars[0], connected_vars[1]);
    let other_var = if target_var == from_var {
        to_var
    } else {
        from_var
    };

    // Get incoming message from other variable
    let g = graph.graph();
    let messages = graph.messages();

    let mut incoming = Array1::from_elem(domain_size, 1.0);
    for neighbor in g.neighbors_directed(other_var, Direction::Incoming) {
        // Skip the factor we're computing for
        if let Some(msg) = messages.get(&(neighbor, other_var)) {
            incoming = &incoming * msg;
        }
    }

    // Normalize incoming
    let sum: f64 = incoming.iter().sum();
    if sum > 0.0 {
        incoming = &incoming / sum;
    }

    let mut result = Array1::from_elem(domain_size, 0.0);

    // If target is the "from" (preceding) variable
    if target_var == from_var {
        // P(from=i) ∝ sum_j P(to=j) * P(j=i+1|connectivity)
        for i in 0..domain_size {
            let j = i + 1;
            if j < domain_size {
                // Strong connectivity: if to=j, from should be j-1
                result[i] += confidence * incoming[j];
            }
            // Weak prior for non-sequential
            result[i] += (1.0 - confidence) / domain_size as f64;
        }
    } else {
        // Target is the "to" (following) variable
        // P(to=j) ∝ sum_i P(from=i) * P(j=i+1|connectivity)
        for j in 0..domain_size {
            if j > 0 {
                let i = j - 1;
                // Strong connectivity: if from=i, to should be i+1
                result[j] += confidence * incoming[i];
            }
            // Weak prior for non-sequential
            result[j] += (1.0 - confidence) / domain_size as f64;
        }
    }

    result
}

/// Compute message for uniqueness constraint.
fn compute_uniqueness_message(
    connected_vars: &[petgraph::graph::NodeIndex],
    target_var: petgraph::graph::NodeIndex,
    graph: &AssignmentFactorGraph,
    domain_size: usize,
) -> Array1<f64> {
    if connected_vars.len() != 2 {
        return Array1::from_elem(domain_size, 1.0);
    }

    let other_var = if connected_vars[0] == target_var {
        connected_vars[1]
    } else {
        connected_vars[0]
    };

    // Get incoming message from other variable
    let g = graph.graph();
    let messages = graph.messages();

    let mut incoming = Array1::from_elem(domain_size, 1.0);
    for neighbor in g.neighbors_directed(other_var, Direction::Incoming) {
        if let Some(msg) = messages.get(&(neighbor, other_var)) {
            incoming = &incoming * msg;
        }
    }

    // Normalize incoming
    let sum: f64 = incoming.iter().sum();
    if sum > 0.0 {
        incoming = &incoming / sum;
    }

    // Uniqueness: penalize assigning to same position as other variable
    // P(target=i) ∝ 1 - P(other=i)
    let mut result = Array1::from_elem(domain_size, 0.0);
    for i in 0..domain_size {
        result[i] = 1.0 - incoming[i] * 0.9; // Soft constraint
    }

    result
}

/// Extract final assignments from marginals with uniqueness enforcement.
///
/// Uses greedy assignment: assign highest-confidence spin systems first,
/// then mark those residues as taken for subsequent assignments.
fn extract_assignments(graph: &AssignmentFactorGraph) -> Vec<MappingResult> {
    let g = graph.graph();
    let messages = graph.messages();
    let domain_size = graph.domain_size();

    // First, compute marginals for all spin systems
    let mut marginals: Vec<(Uuid, Array1<f64>)> = Vec::new();

    for (&spin_system_id, &var_idx) in graph.var_indices() {
        let mut marginal = Array1::from_elem(domain_size, 1.0);

        for neighbor in g.neighbors_directed(var_idx, Direction::Incoming) {
            if let Some(msg) = messages.get(&(neighbor, var_idx)) {
                marginal = &marginal * msg;
            }
        }

        // Normalize
        let sum: f64 = marginal.iter().sum();
        if sum > 0.0 {
            marginal = &marginal / sum;
        }

        marginals.push((spin_system_id, marginal));
    }

    // Greedy assignment: process in order of highest confidence
    let mut results = Vec::new();
    let mut assigned_residues: std::collections::HashSet<i32> = std::collections::HashSet::new();

    // Sort by max probability (highest confidence first)
    let mut assignments: Vec<(Uuid, Array1<f64>, usize, f64)> = marginals
        .into_iter()
        .map(|(id, marginal)| {
            let (best_idx, &best_prob) = marginal
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            (id, marginal, best_idx, best_prob)
        })
        .collect();

    assignments.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

    for (spin_system_id, marginal, _, _) in assignments {
        // Find best AVAILABLE assignment
        let best_available = marginal
            .iter()
            .enumerate()
            .filter(|(i, _)| !assigned_residues.contains(&((*i + 1) as i32)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());

        if let Some((best_idx, &best_prob)) = best_available {
            let assigned_residue = (best_idx + 1) as i32;
            assigned_residues.insert(assigned_residue);

            // Collect alternatives above threshold (excluding assigned residues)
            let alternatives: Vec<(i32, f64)> = marginal
                .iter()
                .enumerate()
                .filter(|(i, &p)| {
                    let res = (*i + 1) as i32;
                    res != assigned_residue && p > 0.05 && !assigned_residues.contains(&res)
                })
                .map(|(i, &p)| ((i + 1) as i32, p))
                .collect();

            results.push(MappingResult {
                spin_system_id,
                assigned_residue,
                confidence: best_prob,
                alternatives,
            });
        }
    }

    // Sort by confidence (highest first)
    results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    results
}

/// Apply mapping results to spin systems.
pub fn apply_mapping(spin_systems: &mut [SpinSystem], results: &[MappingResult]) {
    for result in results {
        if let Some(ss) = spin_systems.iter_mut().find(|ss| ss.id == result.spin_system_id) {
            ss.assigned_residue = Some(result.assigned_residue);
            ss.assignment_confidence = result.confidence;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{NOEType, SequentialLink, UnlabeledPeak};
    use crate::inference::assignment_graph::build_assignment_graph;

    fn make_test_spin_systems() -> Vec<SpinSystem> {
        let anchor1 = UnlabeledPeak::hsqc_15n(123.5, 8.21, 1000.0);
        let anchor2 = UnlabeledPeak::hsqc_15n(119.8, 8.05, 900.0);

        let mut ss1 = SpinSystem::new(&anchor1);
        ss1.amino_acid_scores.insert("ALA".to_string(), 0.9);
        ss1.amino_acid_scores.insert("VAL".to_string(), 0.05);

        let mut ss2 = SpinSystem::new(&anchor2);
        ss2.amino_acid_scores.insert("CYS".to_string(), 0.85);
        ss2.amino_acid_scores.insert("SER".to_string(), 0.1);

        vec![ss1, ss2]
    }

    #[test]
    fn test_simple_mapping() {
        let spin_systems = make_test_spin_systems();
        let sequence = "AC"; // A=ALA at pos 1, C=CYS at pos 2

        let links = vec![SequentialLink::new(
            spin_systems[0].id,
            spin_systems[1].id,
            NOEType::DAlphaN,
            0.9,
        )];

        let mut graph = build_assignment_graph(&spin_systems, &links, sequence);
        let params = MappingParams::default();

        let results = map_to_sequence(&mut graph, &params);

        assert_eq!(results.len(), 2);

        // Find results by spin system
        let ss1_result = results
            .iter()
            .find(|r| r.spin_system_id == spin_systems[0].id)
            .unwrap();
        let ss2_result = results
            .iter()
            .find(|r| r.spin_system_id == spin_systems[1].id)
            .unwrap();

        // ss1 has ALA typing, should map to position 1
        assert_eq!(
            ss1_result.assigned_residue, 1,
            "ss1 (ALA) should map to position 1"
        );

        // ss2 has CYS typing, should map to position 2
        assert_eq!(
            ss2_result.assigned_residue, 2,
            "ss2 (CYS) should map to position 2"
        );
    }

    #[test]
    fn test_apply_mapping() {
        let mut spin_systems = make_test_spin_systems();
        let results = vec![
            MappingResult {
                spin_system_id: spin_systems[0].id,
                assigned_residue: 1,
                confidence: 0.9,
                alternatives: vec![],
            },
            MappingResult {
                spin_system_id: spin_systems[1].id,
                assigned_residue: 2,
                confidence: 0.85,
                alternatives: vec![],
            },
        ];

        apply_mapping(&mut spin_systems, &results);

        assert_eq!(spin_systems[0].assigned_residue, Some(1));
        assert_eq!(spin_systems[1].assigned_residue, Some(2));
    }
}
