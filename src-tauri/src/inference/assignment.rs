//! Global assignment algorithms.

use super::belief_propagation::{compute_average_entropy, get_marginals, run_belief_propagation};
use super::factor_graph::FactorGraph;

/// Result of assignment for a single atom.
#[derive(Debug, Clone)]
pub struct AtomAssignment {
    pub atom_id: String,
    pub assigned_shift: f64,
    pub confidence: f64,
    pub alternatives: Vec<(f64, f64)>, // (shift, probability)
}

/// Parameters for adaptive tolerance optimization.
#[derive(Debug, Clone)]
pub struct AdaptiveToleranceParams {
    /// Target normalized entropy (0.0 = fully peaked, 1.0 = uniform).
    /// Default: 0.15 means we want ~15% of maximum entropy.
    pub target_normalized_entropy: f64,
    /// Maximum number of tolerance adjustment rounds.
    pub max_rounds: usize,
    /// Tolerance adjustment factor per round.
    /// Values < 1.0 tighten tolerance when entropy is too high.
    pub adjustment_factor: f64,
    /// Minimum tolerance to prevent numerical issues.
    pub min_tolerance: f64,
}

impl Default for AdaptiveToleranceParams {
    fn default() -> Self {
        Self {
            target_normalized_entropy: 0.15, // ~15% of max entropy = fairly confident
            max_rounds: 5,
            adjustment_factor: 0.7, // Reduce tolerance by 30% each round if needed
            min_tolerance: 0.01,    // Don't go below 0.01 ppm
        }
    }
}

/// Run global assignment with entropy-based adaptive tolerance optimization.
///
/// This iteratively adjusts peak tolerances to achieve a target marginal entropy:
/// 1. Run BP with current tolerances
/// 2. Compute average normalized entropy of marginals
/// 3. If entropy > target, tighten tolerances and repeat
/// 4. Return final assignments when entropy stabilizes
pub fn run_adaptive_assignment(
    graph: &mut FactorGraph,
    max_bp_iterations: usize,
    bp_tolerance: f64,
    params: AdaptiveToleranceParams,
) -> Vec<AtomAssignment> {
    let mut current_round = 0;

    loop {
        // Run belief propagation
        run_belief_propagation(graph, max_bp_iterations, bp_tolerance);

        // Compute entropy
        let (avg_entropy, max_entropy) = compute_average_entropy(graph);
        let normalized_entropy = if max_entropy > 0.0 {
            avg_entropy / max_entropy
        } else {
            0.0
        };

        let current_tolerance = graph.get_average_peak_tolerance().unwrap_or(1.0);

        tracing::info!(
            "Adaptive round {}: entropy={:.3} (norm={:.2}%), tolerance={:.4}",
            current_round,
            avg_entropy,
            normalized_entropy * 100.0,
            current_tolerance
        );

        current_round += 1;

        // Check termination conditions
        if normalized_entropy <= params.target_normalized_entropy {
            tracing::info!(
                "Target entropy reached (norm={:.2}% <= {:.2}%)",
                normalized_entropy * 100.0,
                params.target_normalized_entropy * 100.0
            );
            break;
        }

        if current_round >= params.max_rounds {
            tracing::info!("Max rounds reached, using current tolerances");
            break;
        }

        if current_tolerance * params.adjustment_factor < params.min_tolerance {
            tracing::info!("Min tolerance reached, stopping adjustment");
            break;
        }

        // Tighten tolerances for next round
        graph.scale_peak_tolerances(params.adjustment_factor);
    }

    // Extract final assignments
    extract_assignments(graph)
}

/// Run global assignment on a factor graph (original non-adaptive version).
pub fn run_global_assignment(
    graph: &mut FactorGraph,
    max_iterations: usize,
    tolerance: f64,
) -> Vec<AtomAssignment> {
    // Run belief propagation
    run_belief_propagation(graph, max_iterations, tolerance);

    // Extract assignments
    extract_assignments(graph)
}

/// Extract assignments from marginals.
fn extract_assignments(graph: &FactorGraph) -> Vec<AtomAssignment> {
    let marginals = get_marginals(graph);

    // Convert to assignments
    let g = graph.graph();
    let var_indices = graph.var_indices();

    let mut assignments = Vec::new();

    for (atom_id, marginal) in marginals {
        // Get domain for this variable
        let idx = match var_indices.get(&atom_id) {
            Some(&i) => i,
            None => continue,
        };

        let domain = match &g[idx] {
            super::factor_graph::FactorNode::Variable { domain, .. } => domain,
            _ => continue,
        };

        // Find best assignment
        let (best_idx, &best_prob) = marginal
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));

        let assigned_shift = domain.get(best_idx).copied().unwrap_or(0.0);

        // Calculate confidence as cumulative probability within 0.1 ppm of best value
        // This is more meaningful than single-point max probability
        let confidence_window = 0.1; // ppm
        let cumulative_confidence: f64 = marginal
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                let shift = domain.get(*i).copied().unwrap_or(0.0);
                (shift - assigned_shift).abs() < confidence_window
            })
            .map(|(_, &p)| p)
            .sum();

        // Collect alternatives above threshold (distinct peaks only)
        let alternatives: Vec<(f64, f64)> = marginal
            .iter()
            .enumerate()
            .filter(|(i, &p)| {
                let shift = domain.get(*i).copied().unwrap_or(0.0);
                // Must be outside confidence window and significant
                (shift - assigned_shift).abs() >= confidence_window && p > 0.01
            })
            .map(|(i, &p)| (domain[i], p))
            .collect();

        assignments.push(AtomAssignment {
            atom_id,
            assigned_shift,
            confidence: cumulative_confidence,
            alternatives,
        });
    }

    assignments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_assignment() {
        let mut graph = FactorGraph::new();

        // Setup simple graph
        graph.add_chemical_shift_variable("A1_H", vec![7.0, 7.5, 8.0, 8.5, 9.0]);
        graph.add_bmrb_prior_factor("A1_H", "H", 8.2, 0.6);
        graph.add_peak_factor(vec!["A1_H"], vec![8.3], vec![0.1]);

        let assignments = run_global_assignment(&mut graph, 50, 1e-6);

        assert_eq!(assignments.len(), 1);

        let h_assignment = &assignments[0];
        assert_eq!(h_assignment.atom_id, "A1_H");
        // Should assign to 8.0 or 8.5 (closest to peak at 8.3)
        assert!(h_assignment.assigned_shift >= 8.0 && h_assignment.assigned_shift <= 8.5);
    }
}
