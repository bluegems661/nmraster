//! Factor graph construction for global NMR assignment.

use ndarray::Array1;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;

/// Node in the factor graph.
#[derive(Clone, Debug)]
pub enum FactorNode {
    /// Variable node (chemical shift assignment)
    Variable {
        id: String,
        domain: Vec<f64>,
    },
    /// Factor node (constraint potential)
    Factor {
        potential: FactorPotential,
        connected_vars: Vec<NodeIndex>,
    },
}

/// Potential functions for factor nodes.
#[derive(Clone, Debug)]
pub enum FactorPotential {
    /// Prior from BMRB statistics
    ChemicalShiftPrior {
        mean: f64,
        std: f64,
        atom_type: String,
    },
    /// Consistency with observed peak position
    PeakConsistency {
        peak_position: f64,
        tolerance: f64,
    },
    /// Sequential connectivity constraint
    SequentialConnectivity {
        expected_shift_difference: f64,
        tolerance: f64,
    },
    /// NOE distance constraint
    NOEDistance {
        distance_bounds: (f64, f64),
    },
}

/// Factor graph for belief propagation.
pub struct FactorGraph {
    graph: DiGraph<FactorNode, f64>,
    var_indices: HashMap<String, NodeIndex>,
    messages: HashMap<(NodeIndex, NodeIndex), Array1<f64>>,
}

impl FactorGraph {
    /// Create a new empty factor graph.
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            var_indices: HashMap::new(),
            messages: HashMap::new(),
        }
    }

    /// Add a chemical shift variable node.
    pub fn add_chemical_shift_variable(
        &mut self,
        atom_id: &str,
        possible_shifts: Vec<f64>,
    ) -> NodeIndex {
        let node = FactorNode::Variable {
            id: atom_id.to_string(),
            domain: possible_shifts,
        };
        let idx = self.graph.add_node(node);
        self.var_indices.insert(atom_id.to_string(), idx);
        idx
    }

    /// Add a BMRB prior factor.
    pub fn add_bmrb_prior_factor(
        &mut self,
        atom_id: &str,
        atom_type: &str,
        mean: f64,
        std: f64,
    ) {
        let var_idx = match self.var_indices.get(atom_id) {
            Some(&idx) => idx,
            None => return,
        };

        let factor = FactorNode::Factor {
            potential: FactorPotential::ChemicalShiftPrior {
                mean,
                std,
                atom_type: atom_type.to_string(),
            },
            connected_vars: vec![var_idx],
        };

        let factor_idx = self.graph.add_node(factor);
        self.graph.add_edge(var_idx, factor_idx, 1.0);
        self.graph.add_edge(factor_idx, var_idx, 1.0);
    }

    /// Add a peak consistency factor.
    pub fn add_peak_factor(
        &mut self,
        atom_ids: Vec<&str>,
        peak_positions: Vec<f64>,
        tolerances: Vec<f64>,
    ) {
        for (i, (atom_id, (&pos, &tol))) in atom_ids
            .iter()
            .zip(peak_positions.iter().zip(tolerances.iter()))
            .enumerate()
        {
            let var_idx = match self.var_indices.get(*atom_id) {
                Some(&idx) => idx,
                None => continue,
            };

            let factor = FactorNode::Factor {
                potential: FactorPotential::PeakConsistency {
                    peak_position: pos,
                    tolerance: tol,
                },
                connected_vars: vec![var_idx],
            };

            let factor_idx = self.graph.add_node(factor);
            self.graph.add_edge(var_idx, factor_idx, 1.0);
            self.graph.add_edge(factor_idx, var_idx, 1.0);
        }
    }

    /// Get the number of variable nodes.
    pub fn num_variables(&self) -> usize {
        self.var_indices.len()
    }

    /// Get the number of factor nodes.
    pub fn num_factors(&self) -> usize {
        self.graph.node_count() - self.var_indices.len()
    }

    /// Get the underlying graph.
    pub fn graph(&self) -> &DiGraph<FactorNode, f64> {
        &self.graph
    }

    /// Get variable indices.
    pub fn var_indices(&self) -> &HashMap<String, NodeIndex> {
        &self.var_indices
    }

    /// Get messages.
    pub fn messages(&self) -> &HashMap<(NodeIndex, NodeIndex), Array1<f64>> {
        &self.messages
    }

    /// Get mutable messages.
    pub fn messages_mut(&mut self) -> &mut HashMap<(NodeIndex, NodeIndex), Array1<f64>> {
        &mut self.messages
    }

    /// Get the domain for a variable by ID.
    pub fn get_domain(&self, atom_id: &str) -> Option<&Vec<f64>> {
        let idx = self.var_indices.get(atom_id)?;
        match &self.graph[*idx] {
            FactorNode::Variable { domain, .. } => Some(domain),
            _ => None,
        }
    }

    /// Calculate domain spacing for a variable.
    ///
    /// Returns the average spacing between domain points, which is useful
    /// for calculating appropriate tolerance values for peak factors.
    pub fn get_domain_spacing(&self, atom_id: &str) -> Option<f64> {
        let domain = self.get_domain(atom_id)?;
        if domain.len() < 2 {
            return None;
        }
        let min = domain.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = domain.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Some((max - min) / (domain.len() - 1) as f64)
    }

    /// Scale all peak consistency factor tolerances by a multiplier.
    ///
    /// This is used for adaptive tolerance optimization based on marginal entropy.
    /// A multiplier < 1.0 tightens tolerances (more peaked marginals).
    /// A multiplier > 1.0 loosens tolerances (flatter marginals).
    pub fn scale_peak_tolerances(&mut self, multiplier: f64) {
        for node in self.graph.node_weights_mut() {
            if let FactorNode::Factor { potential, .. } = node {
                if let FactorPotential::PeakConsistency { tolerance, .. } = potential {
                    *tolerance *= multiplier;
                }
            }
        }
    }

    /// Set all peak consistency factor tolerances to a specific value.
    pub fn set_peak_tolerances(&mut self, new_tolerance: f64) {
        for node in self.graph.node_weights_mut() {
            if let FactorNode::Factor { potential, .. } = node {
                if let FactorPotential::PeakConsistency { tolerance, .. } = potential {
                    *tolerance = new_tolerance;
                }
            }
        }
    }

    /// Get the current average peak tolerance.
    pub fn get_average_peak_tolerance(&self) -> Option<f64> {
        let mut sum = 0.0;
        let mut count = 0;

        for node in self.graph.node_weights() {
            if let FactorNode::Factor { potential, .. } = node {
                if let FactorPotential::PeakConsistency { tolerance, .. } = potential {
                    sum += *tolerance;
                    count += 1;
                }
            }
        }

        if count > 0 {
            Some(sum / count as f64)
        } else {
            None
        }
    }
}

impl Default for FactorGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factor_graph_creation() {
        let mut graph = FactorGraph::new();

        graph.add_chemical_shift_variable("A1_CA", vec![50.0, 52.0, 54.0, 56.0, 58.0]);
        graph.add_bmrb_prior_factor("A1_CA", "CA", 54.0, 2.0);

        assert_eq!(graph.num_variables(), 1);
        assert_eq!(graph.num_factors(), 1);
    }

    #[test]
    fn test_peak_factor() {
        let mut graph = FactorGraph::new();

        graph.add_chemical_shift_variable("A1_H", vec![7.0, 7.5, 8.0, 8.5, 9.0]);
        graph.add_chemical_shift_variable("A1_N", vec![115.0, 118.0, 121.0, 124.0]);

        graph.add_peak_factor(
            vec!["A1_H", "A1_N"],
            vec![8.2, 120.5],
            vec![0.1, 0.5],
        );

        assert_eq!(graph.num_variables(), 2);
        assert_eq!(graph.num_factors(), 2);
    }
}
