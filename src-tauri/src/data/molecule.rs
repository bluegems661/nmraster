//! Molecular structure representation using a graph data structure.
//!
//! Uses petgraph for efficient graph operations and traversal.

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// A molecular structure represented as a graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Molecule {
    pub id: Uuid,
    pub name: String,
    /// The underlying graph structure
    #[serde(skip)]
    graph: DiGraph<MoleculeNode, MoleculeEdge>,
    /// Fast lookup from residue sequence code to node index
    #[serde(skip)]
    residue_indices: HashMap<i32, NodeIndex>,
    /// Fast lookup from atom ID to node index
    #[serde(skip)]
    atom_indices: HashMap<Uuid, NodeIndex>,
    /// The amino acid sequence (one-letter codes)
    pub sequence: String,
    /// Chain identifiers
    pub chains: Vec<Chain>,
}

/// A chain within the molecule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chain {
    pub id: Uuid,
    pub chain_code: String,
    pub polymer_type: PolymerType,
    pub start_residue: i32,
    pub end_residue: i32,
}

/// Type of polymer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolymerType {
    Protein,
    DNA,
    RNA,
    Polysaccharide,
    Other,
}

/// Node in the molecular graph - either a residue or an atom.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MoleculeNode {
    Residue(Residue),
    Atom(Atom),
}

/// A residue (amino acid, nucleotide, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Residue {
    pub id: Uuid,
    pub sequence_code: i32,
    pub residue_name: String,      // e.g., "ALA", "GLY"
    pub one_letter_code: char,     // e.g., 'A', 'G'
    pub chain_code: String,
}

impl Residue {
    /// Get standard backbone atom names for this residue type.
    pub fn backbone_atoms(&self) -> Vec<&'static str> {
        vec!["N", "CA", "C", "O", "H", "HA"]
    }

    /// Get standard sidechain atom names including protons.
    /// Uses BMRB naming conventions (HB for methyl groups like ALA, HB2/HB3 for methylenes).
    pub fn sidechain_atoms(&self) -> Vec<&'static str> {
        match self.one_letter_code {
            'G' => vec![], // Glycine has no sidechain
            'A' => vec!["CB", "HB"], // Methyl - BMRB uses "HB" for averaged methyl
            'V' => vec!["CB", "HB", "CG1", "HG1", "CG2", "HG2"],
            'L' => vec!["CB", "HB2", "HB3", "CG", "HG", "CD1", "HD1", "CD2", "HD2"],
            'I' => vec!["CB", "HB", "CG1", "HG12", "HG13", "CG2", "HG2", "CD1", "HD1"],
            'P' => vec!["CB", "HB2", "HB3", "CG", "HG2", "HG3", "CD", "HD2", "HD3"],
            'F' => vec!["CB", "HB2", "HB3", "CG", "CD1", "HD1", "CD2", "HD2", "CE1", "HE1", "CE2", "HE2", "CZ", "HZ"],
            'Y' => vec!["CB", "HB2", "HB3", "CG", "CD1", "HD1", "CD2", "HD2", "CE1", "HE1", "CE2", "HE2", "CZ", "OH", "HH"],
            'W' => vec!["CB", "HB2", "HB3", "CG", "CD1", "HD1", "CD2", "NE1", "HE1", "CE2", "CE3", "HE3", "CZ2", "HZ2", "CZ3", "HZ3", "CH2", "HH2"],
            'S' => vec!["CB", "HB2", "HB3", "OG", "HG"],
            'T' => vec!["CB", "HB", "OG1", "HG1", "CG2", "HG2"],
            'C' => vec!["CB", "HB2", "HB3", "SG", "HG"],
            'M' => vec!["CB", "HB2", "HB3", "CG", "HG2", "HG3", "SD", "CE", "HE"],
            'N' => vec!["CB", "HB2", "HB3", "CG", "OD1", "ND2", "HD21", "HD22"],
            'Q' => vec!["CB", "HB2", "HB3", "CG", "HG2", "HG3", "CD", "OE1", "NE2", "HE21", "HE22"],
            'D' => vec!["CB", "HB2", "HB3", "CG", "OD1", "OD2"],
            'E' => vec!["CB", "HB2", "HB3", "CG", "HG2", "HG3", "CD", "OE1", "OE2"],
            'K' => vec!["CB", "HB2", "HB3", "CG", "HG2", "HG3", "CD", "HD2", "HD3", "CE", "HE2", "HE3", "NZ", "HZ"],
            'R' => vec!["CB", "HB2", "HB3", "CG", "HG2", "HG3", "CD", "HD2", "HD3", "NE", "HE", "CZ", "NH1", "HH11", "HH12", "NH2", "HH21", "HH22"],
            'H' => vec!["CB", "HB2", "HB3", "CG", "ND1", "HD1", "CD2", "HD2", "CE1", "HE1", "NE2", "HE2"],
            _ => vec![],
        }
    }
}

/// An atom within a residue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Atom {
    pub id: Uuid,
    pub atom_name: String,         // e.g., "CA", "HN"
    pub element: Element,
    pub isotope_number: Option<u8>, // e.g., 13 for 13C
    pub residue_id: Uuid,
}

/// Chemical elements relevant to NMR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Element {
    H,
    C,
    N,
    O,
    S,
    P,
    F,
    Other,
}

/// Edge in the molecular graph.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MoleculeEdge {
    /// Covalent bond between atoms
    Bond(BondType),
    /// Sequential connectivity (residue i to i+1)
    Sequence,
    /// Atom belongs to residue
    Contains,
}

/// Type of covalent bond.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BondType {
    Single,
    Double,
    Triple,
    Aromatic,
    Peptide,
}

impl Molecule {
    /// Create a new empty molecule.
    pub fn new(name: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            graph: DiGraph::new(),
            residue_indices: HashMap::new(),
            atom_indices: HashMap::new(),
            sequence: String::new(),
            chains: Vec::new(),
        }
    }

    /// Create a molecule from a one-letter amino acid sequence.
    pub fn from_sequence(name: &str, sequence: &str, chain_code: &str) -> Self {
        let mut mol = Self::new(name);
        mol.sequence = sequence.to_string();

        // Create chain
        let chain = Chain {
            id: Uuid::new_v4(),
            chain_code: chain_code.to_string(),
            polymer_type: PolymerType::Protein,
            start_residue: 1,
            end_residue: sequence.len() as i32,
        };
        mol.chains.push(chain);

        // Create residues
        let mut prev_residue_idx: Option<NodeIndex> = None;

        for (i, one_letter) in sequence.chars().enumerate() {
            let seq_code = (i + 1) as i32;
            let residue_name = one_letter_to_three_letter(one_letter);

            let residue = Residue {
                id: Uuid::new_v4(),
                sequence_code: seq_code,
                residue_name,
                one_letter_code: one_letter,
                chain_code: chain_code.to_string(),
            };

            let residue_id = residue.id;
            let backbone_atoms = residue.backbone_atoms();
            let sidechain_atoms = residue.sidechain_atoms();

            // Add residue node
            let residue_idx = mol.graph.add_node(MoleculeNode::Residue(residue));
            mol.residue_indices.insert(seq_code, residue_idx);

            // Add sequential edge from previous residue
            if let Some(prev_idx) = prev_residue_idx {
                mol.graph.add_edge(prev_idx, residue_idx, MoleculeEdge::Sequence);
            }
            prev_residue_idx = Some(residue_idx);

            // Add backbone atoms
            for atom_name in backbone_atoms {
                let element = element_from_atom_name(atom_name);
                let atom = Atom {
                    id: Uuid::new_v4(),
                    atom_name: atom_name.to_string(),
                    element,
                    isotope_number: None,
                    residue_id,
                };
                let atom_idx = mol.graph.add_node(MoleculeNode::Atom(atom.clone()));
                mol.atom_indices.insert(atom.id, atom_idx);
                mol.graph.add_edge(residue_idx, atom_idx, MoleculeEdge::Contains);
            }

            // Add sidechain atoms
            for atom_name in sidechain_atoms {
                let element = element_from_atom_name(atom_name);
                let atom = Atom {
                    id: Uuid::new_v4(),
                    atom_name: atom_name.to_string(),
                    element,
                    isotope_number: None,
                    residue_id,
                };
                let atom_idx = mol.graph.add_node(MoleculeNode::Atom(atom.clone()));
                mol.atom_indices.insert(atom.id, atom_idx);
                mol.graph.add_edge(residue_idx, atom_idx, MoleculeEdge::Contains);
            }
        }

        mol
    }

    /// Get a residue by sequence code.
    pub fn get_residue(&self, seq_code: i32) -> Option<&Residue> {
        self.residue_indices.get(&seq_code).and_then(|&idx| {
            match &self.graph[idx] {
                MoleculeNode::Residue(r) => Some(r),
                _ => None,
            }
        })
    }

    /// Get all atoms for a residue.
    pub fn get_atoms_for_residue(&self, seq_code: i32) -> Vec<&Atom> {
        let Some(&residue_idx) = self.residue_indices.get(&seq_code) else {
            return Vec::new();
        };

        self.graph
            .neighbors_directed(residue_idx, Direction::Outgoing)
            .filter_map(|idx| {
                match &self.graph[idx] {
                    MoleculeNode::Atom(a) => Some(a),
                    _ => None,
                }
            })
            .collect()
    }

    /// Get an atom by its ID.
    pub fn get_atom(&self, atom_id: &Uuid) -> Option<&Atom> {
        self.atom_indices.get(atom_id).and_then(|&idx| {
            match &self.graph[idx] {
                MoleculeNode::Atom(a) => Some(a),
                _ => None,
            }
        })
    }

    /// Get the next residue in sequence.
    pub fn next_residue(&self, seq_code: i32) -> Option<&Residue> {
        self.get_residue(seq_code + 1)
    }

    /// Get the previous residue in sequence.
    pub fn prev_residue(&self, seq_code: i32) -> Option<&Residue> {
        self.get_residue(seq_code - 1)
    }

    /// Get the total number of residues.
    pub fn num_residues(&self) -> usize {
        self.residue_indices.len()
    }

    /// Get the total number of atoms.
    pub fn num_atoms(&self) -> usize {
        self.atom_indices.len()
    }

    /// Iterate over all residues in sequence order.
    pub fn residues(&self) -> impl Iterator<Item = &Residue> {
        let mut seq_codes: Vec<_> = self.residue_indices.keys().copied().collect();
        seq_codes.sort();

        seq_codes.into_iter().filter_map(move |seq_code| {
            self.get_residue(seq_code)
        })
    }
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

/// Determine element from atom name.
fn element_from_atom_name(name: &str) -> Element {
    let first_char = name.chars().next().unwrap_or('X');
    match first_char {
        'H' => Element::H,
        'C' => Element::C,
        'N' => Element::N,
        'O' => Element::O,
        'S' => Element::S,
        'P' => Element::P,
        'F' => Element::F,
        _ => Element::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_molecule_from_sequence() {
        let mol = Molecule::from_sequence("test", "ACDEFG", "A");

        assert_eq!(mol.num_residues(), 6);
        assert!(mol.num_atoms() > 0);

        let ala = mol.get_residue(1).unwrap();
        assert_eq!(ala.one_letter_code, 'A');
        assert_eq!(ala.residue_name, "ALA");

        let gly = mol.get_residue(6).unwrap();
        assert_eq!(gly.one_letter_code, 'G');
    }

    #[test]
    fn test_get_atoms_for_residue() {
        let mol = Molecule::from_sequence("test", "AV", "A");

        let ala_atoms = mol.get_atoms_for_residue(1);
        let atom_names: Vec<_> = ala_atoms.iter().map(|a| a.atom_name.as_str()).collect();

        assert!(atom_names.contains(&"CA"));
        assert!(atom_names.contains(&"CB"));
        assert!(atom_names.contains(&"N"));
    }

    #[test]
    fn test_sequential_navigation() {
        let mol = Molecule::from_sequence("test", "ABC", "A");

        let res2 = mol.get_residue(2).unwrap();
        assert_eq!(res2.one_letter_code, 'B');

        let next = mol.next_residue(2).unwrap();
        assert_eq!(next.one_letter_code, 'C');

        let prev = mol.prev_residue(2).unwrap();
        assert_eq!(prev.one_letter_code, 'A');
    }
}
