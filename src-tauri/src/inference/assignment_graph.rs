//! Factor graph for spin system to sequence assignment.
//!
//! This builds a factor graph where:
//! - Variables represent spin system → residue mappings
//! - Factors encode typing preferences and sequential connectivity
//! - Uniqueness constraints ensure one-to-one mapping

use std::collections::HashMap;

use ndarray::Array1;
use petgraph::graph::{DiGraph, NodeIndex};
use uuid::Uuid;

use crate::data::{SequentialLink, SpinSystem};

/// Node in the assignment factor graph.
#[derive(Clone, Debug)]
pub enum AssignmentNode {
    /// Variable node: which residue does spin system map to?
    SpinSystemVariable {
        spin_system_id: Uuid,
        /// Domain: possible residue sequence codes
        domain: Vec<i32>,
    },
    /// Factor node: constraint on assignments
    AssignmentFactor {
        factor_type: AssignmentFactorType,
        connected_vars: Vec<NodeIndex>,
    },
}

/// Types of factors in the assignment graph.
#[derive(Clone, Debug)]
pub enum AssignmentFactorType {
    /// Prior from amino acid typing (spin system pattern matches residue type)
    TypingPrior {
        /// Scores indexed by domain position
        scores: Vec<f64>,
    },
    /// Sequential connectivity (spin system i precedes spin system j)
    SequentialConnectivity {
        confidence: f64,
    },
    /// Uniqueness constraint (no two spin systems can map to same residue)
    Uniqueness,
}

/// Factor graph for spin system assignment.
pub struct AssignmentFactorGraph {
    graph: DiGraph<AssignmentNode, f64>,
    /// Map from spin system ID to variable node index
    var_indices: HashMap<Uuid, NodeIndex>,
    /// Messages for belief propagation
    messages: HashMap<(NodeIndex, NodeIndex), Array1<f64>>,
    /// Domain size (number of residues)
    domain_size: usize,
    /// Sequence for reference
    sequence: String,
}

impl AssignmentFactorGraph {
    /// Create a new assignment factor graph.
    ///
    /// # Arguments
    /// * `sequence` - The protein sequence (one-letter codes)
    pub fn new(sequence: &str) -> Self {
        Self {
            graph: DiGraph::new(),
            var_indices: HashMap::new(),
            messages: HashMap::new(),
            domain_size: sequence.len(),
            sequence: sequence.to_string(),
        }
    }

    /// Add a spin system variable node.
    ///
    /// The domain is all possible residue positions (1 to sequence_length).
    pub fn add_spin_system_variable(&mut self, spin_system: &SpinSystem) -> NodeIndex {
        let domain: Vec<i32> = (1..=self.domain_size as i32).collect();

        let node = AssignmentNode::SpinSystemVariable {
            spin_system_id: spin_system.id,
            domain,
        };

        let idx = self.graph.add_node(node);
        self.var_indices.insert(spin_system.id, idx);
        idx
    }

    /// Add amino acid typing factor for a spin system.
    ///
    /// This encodes the prior probability of the spin system mapping to each
    /// residue based on amino acid type matching.
    pub fn add_typing_factor(&mut self, spin_system: &SpinSystem) {
        let var_idx = match self.var_indices.get(&spin_system.id) {
            Some(&idx) => idx,
            None => return,
        };

        // Build scores for each position in sequence
        let mut scores = Vec::with_capacity(self.domain_size);

        for (_i, aa_char) in self.sequence.chars().enumerate() {
            let aa_3letter = one_letter_to_three_letter(aa_char);

            // Get typing score for this amino acid type
            let score = spin_system
                .amino_acid_scores
                .get(&aa_3letter)
                .copied()
                .unwrap_or(0.01); // Small prior for unknown types

            scores.push(score);
        }

        let factor = AssignmentNode::AssignmentFactor {
            factor_type: AssignmentFactorType::TypingPrior { scores },
            connected_vars: vec![var_idx],
        };

        let factor_idx = self.graph.add_node(factor);
        self.graph.add_edge(var_idx, factor_idx, 1.0);
        self.graph.add_edge(factor_idx, var_idx, 1.0);
    }

    /// Add sequential connectivity factor between two spin systems.
    ///
    /// This encodes that if spin system A maps to residue i, spin system B
    /// should map to residue i+1.
    pub fn add_sequential_factor(&mut self, link: &SequentialLink) {
        let var_from = match self.var_indices.get(&link.from_spin_system) {
            Some(&idx) => idx,
            None => return,
        };
        let var_to = match self.var_indices.get(&link.to_spin_system) {
            Some(&idx) => idx,
            None => return,
        };

        let factor = AssignmentNode::AssignmentFactor {
            factor_type: AssignmentFactorType::SequentialConnectivity {
                confidence: link.confidence,
            },
            connected_vars: vec![var_from, var_to],
        };

        let factor_idx = self.graph.add_node(factor);
        self.graph.add_edge(var_from, factor_idx, 1.0);
        self.graph.add_edge(var_to, factor_idx, 1.0);
        self.graph.add_edge(factor_idx, var_from, 1.0);
        self.graph.add_edge(factor_idx, var_to, 1.0);
    }

    /// Add uniqueness factors between all pairs of spin systems.
    ///
    /// This ensures no two spin systems can map to the same residue.
    pub fn add_uniqueness_factors(&mut self) {
        let var_ids: Vec<Uuid> = self.var_indices.keys().cloned().collect();

        for i in 0..var_ids.len() {
            for j in (i + 1)..var_ids.len() {
                let var_i = self.var_indices[&var_ids[i]];
                let var_j = self.var_indices[&var_ids[j]];

                let factor = AssignmentNode::AssignmentFactor {
                    factor_type: AssignmentFactorType::Uniqueness,
                    connected_vars: vec![var_i, var_j],
                };

                let factor_idx = self.graph.add_node(factor);
                self.graph.add_edge(var_i, factor_idx, 1.0);
                self.graph.add_edge(var_j, factor_idx, 1.0);
                self.graph.add_edge(factor_idx, var_i, 1.0);
                self.graph.add_edge(factor_idx, var_j, 1.0);
            }
        }
    }

    /// Get the underlying graph.
    pub fn graph(&self) -> &DiGraph<AssignmentNode, f64> {
        &self.graph
    }

    /// Get variable indices.
    pub fn var_indices(&self) -> &HashMap<Uuid, NodeIndex> {
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

    /// Get domain size.
    pub fn domain_size(&self) -> usize {
        self.domain_size
    }

    /// Get number of variables.
    pub fn num_variables(&self) -> usize {
        self.var_indices.len()
    }

    /// Get number of factors.
    pub fn num_factors(&self) -> usize {
        self.graph.node_count() - self.var_indices.len()
    }
}

/// Build a complete assignment factor graph.
pub fn build_assignment_graph(
    spin_systems: &[SpinSystem],
    links: &[SequentialLink],
    sequence: &str,
) -> AssignmentFactorGraph {
    let mut graph = AssignmentFactorGraph::new(sequence);

    // Add variables for each spin system
    for ss in spin_systems {
        graph.add_spin_system_variable(ss);
    }

    // Add typing factors
    for ss in spin_systems {
        graph.add_typing_factor(ss);
    }

    // Add sequential factors
    for link in links {
        graph.add_sequential_factor(link);
    }

    // Add uniqueness constraints
    graph.add_uniqueness_factors();

    tracing::info!(
        "Built assignment graph: {} variables, {} factors for {} residues",
        graph.num_variables(),
        graph.num_factors(),
        sequence.len()
    );

    graph
}

/// Convert one-letter amino acid code to three-letter code.
fn one_letter_to_three_letter(one: char) -> String {
    match one.to_ascii_uppercase() {
        'A' => "ALA",
        'C' => "CYS",
        'D' => "ASP",
        'E' => "GLU",
        'F' => "PHE",
        'G' => "GLY",
        'H' => "HIS",
        'I' => "ILE",
        'K' => "LYS",
        'L' => "LEU",
        'M' => "MET",
        'N' => "ASN",
        'P' => "PRO",
        'Q' => "GLN",
        'R' => "ARG",
        'S' => "SER",
        'T' => "THR",
        'V' => "VAL",
        'W' => "TRP",
        'Y' => "TYR",
        _ => "UNK",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{NOEType, UnlabeledPeak};

    fn make_typed_spin_systems() -> Vec<SpinSystem> {
        let anchor1 = UnlabeledPeak::hsqc_15n(123.5, 8.21, 1000.0);
        let anchor2 = UnlabeledPeak::hsqc_15n(119.8, 8.05, 900.0);

        let mut ss1 = SpinSystem::new(&anchor1);
        ss1.amino_acid_scores.insert("ALA".to_string(), 0.8);
        ss1.amino_acid_scores.insert("VAL".to_string(), 0.1);

        let mut ss2 = SpinSystem::new(&anchor2);
        ss2.amino_acid_scores.insert("CYS".to_string(), 0.7);
        ss2.amino_acid_scores.insert("SER".to_string(), 0.2);

        vec![ss1, ss2]
    }

    #[test]
    fn test_build_assignment_graph() {
        let spin_systems = make_typed_spin_systems();
        let sequence = "ACDEFG";

        let links = vec![SequentialLink::new(
            spin_systems[0].id,
            spin_systems[1].id,
            NOEType::DAlphaN,
            0.9,
        )];

        let graph = build_assignment_graph(&spin_systems, &links, sequence);

        assert_eq!(graph.num_variables(), 2);
        // Factors: 2 typing + 1 sequential + 1 uniqueness = 4
        assert_eq!(graph.num_factors(), 4);
        assert_eq!(graph.domain_size(), 6);
    }

    #[test]
    fn test_typing_factor_scores() {
        let spin_systems = make_typed_spin_systems();
        let sequence = "AC"; // A=ALA, C=CYS

        let mut graph = AssignmentFactorGraph::new(sequence);
        graph.add_spin_system_variable(&spin_systems[0]);
        graph.add_typing_factor(&spin_systems[0]);

        // Check that typing factor was added
        assert_eq!(graph.num_factors(), 1);

        // The factor should have higher score for position 0 (ALA) than position 1 (CYS)
        let factor_idx = graph
            .graph()
            .node_indices()
            .find(|&idx| {
                matches!(
                    &graph.graph()[idx],
                    AssignmentNode::AssignmentFactor { .. }
                )
            })
            .unwrap();

        if let AssignmentNode::AssignmentFactor {
            factor_type: AssignmentFactorType::TypingPrior { scores },
            ..
        } = &graph.graph()[factor_idx]
        {
            // ss1 has ALA score 0.8, CYS score not set (default 0.01)
            assert!(scores[0] > scores[1], "ALA score should be higher than CYS");
        } else {
            panic!("Expected typing factor");
        }
    }
}
