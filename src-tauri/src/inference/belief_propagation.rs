//! Belief propagation algorithm for factor graphs.

use ndarray::Array1;
use petgraph::Direction;
use std::collections::HashMap;

use super::factor_graph::{FactorGraph, FactorNode, FactorPotential};
use petgraph::graph::NodeIndex;

/// Run belief propagation on a factor graph.
pub fn run_belief_propagation(
    graph: &mut FactorGraph,
    max_iterations: usize,
    tolerance: f64,
) -> bool {
    // Initialize messages
    initialize_messages(graph);

    // Iterate until convergence or max iterations
    for iteration in 0..max_iterations {
        let max_delta = propagate_one_step(graph);

        if max_delta < tolerance {
            tracing::info!("BP converged at iteration {}", iteration);
            return true;
        }
    }

    tracing::warn!("BP did not converge after {} iterations", max_iterations);
    false
}

/// Initialize all messages to uniform distributions.
fn initialize_messages(graph: &mut FactorGraph) {
    // Collect edge info first
    let edges: Vec<_> = {
        let g = graph.graph();
        g.edge_indices()
            .filter_map(|edge| {
                let (from, to) = g.edge_endpoints(edge)?;
                let domain_size = get_domain_size_internal(g, from, to);
                if domain_size > 0 {
                    Some(((from, to), domain_size))
                } else {
                    None
                }
            })
            .collect()
    };

    // Now insert messages
    for ((from, to), domain_size) in edges {
        let uniform = Array1::from_elem(domain_size, 0.0); // Log-space uniform
        graph.messages_mut().insert((from, to), uniform);
    }
}

/// Get the domain size for message between two nodes (internal helper).
fn get_domain_size_internal(
    g: &petgraph::Graph<FactorNode, f64>,
    from: NodeIndex,
    to: NodeIndex,
) -> usize {
    // Find the variable node involved
    let var_node = if matches!(g[from], FactorNode::Variable { .. }) {
        from
    } else if matches!(g[to], FactorNode::Variable { .. }) {
        to
    } else {
        return 0;
    };

    match &g[var_node] {
        FactorNode::Variable { domain, .. } => domain.len(),
        _ => 0,
    }
}

/// Get the domain size for message between two nodes.
fn get_domain_size(
    g: &petgraph::Graph<FactorNode, f64>,
    var_indices: &HashMap<String, NodeIndex>,
    from: NodeIndex,
    to: NodeIndex,
) -> usize {
    // Find the variable node involved
    let var_node = if matches!(g[from], FactorNode::Variable { .. }) {
        from
    } else if matches!(g[to], FactorNode::Variable { .. }) {
        to
    } else {
        return 0;
    };

    match &g[var_node] {
        FactorNode::Variable { domain, .. } => domain.len(),
        _ => 0,
    }
}

/// Perform one iteration of message passing.
fn propagate_one_step(graph: &mut FactorGraph) -> f64 {
    let mut max_delta = 0.0f64;
    let g = graph.graph().clone();

    for edge in g.edge_indices() {
        let (from, to) = g.edge_endpoints(edge).unwrap();
        let new_message = compute_message(&g, graph.messages(), from, to);

        let key = (from, to);
        if let Some(old_message) = graph.messages().get(&key) {
            let delta = (&new_message - old_message).mapv(|x| x.abs()).sum();
            max_delta = max_delta.max(delta);
        }
        graph.messages_mut().insert(key, new_message);
    }

    max_delta
}

/// Compute message from one node to another.
fn compute_message(
    g: &petgraph::Graph<FactorNode, f64>,
    messages: &HashMap<(NodeIndex, NodeIndex), Array1<f64>>,
    from: NodeIndex,
    to: NodeIndex,
) -> Array1<f64> {
    match &g[from] {
        FactorNode::Variable { domain, .. } => {
            // Variable to factor: product of incoming messages except from target
            let mut msg = Array1::zeros(domain.len());
            for neighbor in g.neighbors_directed(from, Direction::Outgoing) {
                if neighbor != to {
                    if let Some(incoming) = messages.get(&(neighbor, from)) {
                        msg += incoming;
                    }
                }
            }
            normalize_log_message(msg)
        }
        FactorNode::Factor { potential, .. } => {
            // Factor to variable: marginalize potential
            compute_factor_message(g, potential, to)
        }
    }
}

/// Compute message from factor to variable.
fn compute_factor_message(
    g: &petgraph::Graph<FactorNode, f64>,
    potential: &FactorPotential,
    target: NodeIndex,
) -> Array1<f64> {
    let domain = match &g[target] {
        FactorNode::Variable { domain, .. } => domain,
        _ => return Array1::zeros(1),
    };

    let mut message = Array1::zeros(domain.len());

    match potential {
        FactorPotential::ChemicalShiftPrior { mean, std, .. } => {
            for (i, &value) in domain.iter().enumerate() {
                let log_prob = -0.5 * ((value - mean) / std).powi(2);
                message[i] = log_prob;
            }
        }
        FactorPotential::PeakConsistency { peak_position, tolerance } => {
            for (i, &value) in domain.iter().enumerate() {
                let diff = (value - peak_position).abs();
                // Use very tight Gaussian with steep falloff outside tolerance
                // This makes observed peaks strongly constrain the assignment
                let log_prob = if diff < *tolerance * 3.0 {
                    // Steeper Gaussian: (diff/tolerance)^2 falls off much faster
                    -2.0 * (diff / tolerance).powi(2)
                } else {
                    -50.0 // Very strong penalty - effectively impossible
                };
                message[i] = log_prob;
            }
        }
        FactorPotential::SequentialConnectivity { expected_shift_difference, tolerance } => {
            // Simplified: just apply prior based on expected difference
            for (i, &value) in domain.iter().enumerate() {
                let diff = (value - expected_shift_difference).abs();
                message[i] = -0.5 * (diff / tolerance).powi(2);
            }
        }
        FactorPotential::NOEDistance { distance_bounds } => {
            // NOE factors typically connect to structural features
            // For now, uniform distribution
            for i in 0..domain.len() {
                message[i] = 0.0;
            }
        }
    }

    normalize_log_message(message)
}

/// Normalize a message in log space.
fn normalize_log_message(mut msg: Array1<f64>) -> Array1<f64> {
    let max_val = msg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val.is_finite() {
        msg -= max_val;
    }
    msg
}

/// Extract marginal probabilities for all variables.
pub fn get_marginals(graph: &FactorGraph) -> HashMap<String, Array1<f64>> {
    let g = graph.graph();
    let messages = graph.messages();
    let mut marginals = HashMap::new();

    for (id, &idx) in graph.var_indices() {
        if let FactorNode::Variable { domain, .. } = &g[idx] {
            let mut marginal = Array1::zeros(domain.len());
            for neighbor in g.neighbors_directed(idx, Direction::Incoming) {
                if let Some(msg) = messages.get(&(neighbor, idx)) {
                    marginal += msg;
                }
            }
            marginals.insert(id.clone(), softmax(&marginal));
        }
    }

    marginals
}

/// Convert log probabilities to probabilities using softmax.
fn softmax(log_probs: &Array1<f64>) -> Array1<f64> {
    let max_val = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals = log_probs.mapv(|x| (x - max_val).exp());
    let sum = exp_vals.sum();
    if sum > 0.0 {
        exp_vals / sum
    } else {
        Array1::from_elem(log_probs.len(), 1.0 / log_probs.len() as f64)
    }
}

/// Compute Shannon entropy of a probability distribution.
/// Returns entropy in bits (log base 2).
/// Higher entropy = more uncertainty (flatter distribution).
/// Lower entropy = more certainty (peaked distribution).
pub fn compute_entropy(probs: &Array1<f64>) -> f64 {
    let epsilon = 1e-10;
    probs
        .iter()
        .filter(|&&p| p > epsilon)
        .map(|&p| -p * p.log2())
        .sum()
}

/// Compute average entropy across all marginals.
/// Returns (average_entropy, max_possible_entropy).
pub fn compute_average_entropy(graph: &FactorGraph) -> (f64, f64) {
    let marginals = get_marginals(graph);

    if marginals.is_empty() {
        return (0.0, 0.0);
    }

    let mut total_entropy = 0.0;
    let mut total_max_entropy = 0.0;

    for (_id, marginal) in &marginals {
        let entropy = compute_entropy(marginal);
        let max_entropy = (marginal.len() as f64).log2(); // Uniform distribution entropy
        total_entropy += entropy;
        total_max_entropy += max_entropy;
    }

    let n = marginals.len() as f64;
    (total_entropy / n, total_max_entropy / n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_belief_propagation_simple() {
        let mut graph = FactorGraph::new();

        // Add a variable with domain
        graph.add_chemical_shift_variable("A1_CA", vec![50.0, 52.0, 54.0, 56.0, 58.0]);

        // Add BMRB prior centered at 54
        graph.add_bmrb_prior_factor("A1_CA", "CA", 54.0, 2.0);

        // Run BP
        let converged = run_belief_propagation(&mut graph, 10, 1e-6);
        assert!(converged);

        // Get marginals
        let marginals = get_marginals(&graph);
        let ca_marginal = &marginals["A1_CA"];

        // Maximum should be at index 2 (value 54.0)
        let max_idx = ca_marginal
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(max_idx, 2);
    }

    #[test]
    fn test_softmax() {
        let log_probs = Array1::from_vec(vec![-1.0, 0.0, -2.0]);
        let probs = softmax(&log_probs);

        // Sum should be 1
        assert!((probs.sum() - 1.0).abs() < 1e-10);

        // Index 1 (value 0.0) should have highest probability
        assert!(probs[1] > probs[0]);
        assert!(probs[1] > probs[2]);
    }
}
