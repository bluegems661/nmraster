//! Residue topology definitions for NMR assignment.
//!
//! This module defines the physical structure of amino acids - what atoms exist,
//! which are stereo-equivalent, and which have special NH groups. These are
//! HARD PHYSICAL CONSTRAINTS, not soft preferences.
//!
//! Key principle: The sequence defines exactly which atoms exist at each position.
//! If a residue is Isoleucine, it HAS CG1 and CG2 - this is not a probability.


/// Complete topology definition for an amino acid residue.
#[derive(Debug, Clone)]
pub struct ResidueTopology {
    pub one_letter: char,
    pub three_letter: &'static str,

    /// Backbone heavy atoms (N, CA, C, O)
    pub backbone_heavy: &'static [&'static str],

    /// Backbone protons (H, HA or HA2/HA3 for Gly)
    pub backbone_protons: &'static [&'static str],

    /// Sidechain carbon atoms
    pub sidechain_carbons: &'static [&'static str],

    /// Sidechain protons
    pub sidechain_protons: &'static [&'static str],

    /// Sidechain heteroatoms (N, O, S)
    pub sidechain_hetero: &'static [&'static str],

    /// Does this residue have a backbone amide NH? (false for Pro)
    pub has_backbone_nh: bool,

    /// Sidechain NH/NH2 groups that appear in 15N-HSQC
    pub sidechain_nh_groups: &'static [SidechainNH],

    /// Stereo-equivalent carbon pairs (for evaluation - assignments to either count as correct)
    /// Note: Ile CG1/CG2 are NOT equivalent (different chemical types)
    pub stereo_equivalent_carbons: &'static [(&'static str, &'static str)],

    /// Stereo-equivalent proton groups (e.g., HG11/HG12/HG13 within a methyl)
    pub stereo_equivalent_protons: &'static [&'static [&'static str]],
}

/// Sidechain NH or NH2 group that may appear in 15N-HSQC.
#[derive(Debug, Clone, Copy)]
pub struct SidechainNH {
    /// Nitrogen atom name (e.g., "ND2", "NE2", "NE1")
    pub nitrogen: &'static str,
    /// Attached proton names
    pub protons: &'static [&'static str],
    /// Is this an NH2 (amide) or single NH?
    pub is_nh2: bool,
}

impl SidechainNH {
    const fn nh2(nitrogen: &'static str, protons: &'static [&'static str]) -> Self {
        Self { nitrogen, protons, is_nh2: true }
    }

    const fn nh(nitrogen: &'static str, protons: &'static [&'static str]) -> Self {
        Self { nitrogen, protons, is_nh2: false }
    }
}

impl ResidueTopology {
    /// Get all carbon atoms (backbone CA + sidechain)
    pub fn all_carbons(&self) -> Vec<&'static str> {
        let mut carbons = vec!["CA"];
        carbons.extend(self.sidechain_carbons);
        carbons
    }

    /// Get all proton atoms (backbone + sidechain)
    pub fn all_protons(&self) -> Vec<&'static str> {
        let mut protons: Vec<&'static str> = self.backbone_protons.to_vec();
        protons.extend(self.sidechain_protons);
        protons
    }

    /// Get all nitrogen atoms (backbone N + sidechain NH groups)
    pub fn all_nitrogens(&self) -> Vec<&'static str> {
        let mut nitrogens = vec!["N"];
        for nh in self.sidechain_nh_groups {
            nitrogens.push(nh.nitrogen);
        }
        // Add other sidechain nitrogens that aren't in NH groups
        for atom in self.sidechain_hetero {
            if atom.starts_with('N') && !nitrogens.contains(atom) {
                nitrogens.push(atom);
            }
        }
        nitrogens
    }

    /// Check if this residue has a specific atom
    pub fn has_atom(&self, atom_name: &str) -> bool {
        // Backbone
        if self.backbone_heavy.contains(&atom_name) {
            return true;
        }
        if self.backbone_protons.contains(&atom_name) {
            return true;
        }
        // Sidechain
        if self.sidechain_carbons.contains(&atom_name) {
            return true;
        }
        if self.sidechain_protons.contains(&atom_name) {
            return true;
        }
        if self.sidechain_hetero.contains(&atom_name) {
            return true;
        }
        // Special case: CA is always present
        if atom_name == "CA" {
            return true;
        }
        false
    }

    /// Check if two atoms are stereo-equivalent for this residue
    pub fn are_stereo_equivalent(&self, atom1: &str, atom2: &str) -> bool {
        // Check carbon pairs
        for (c1, c2) in self.stereo_equivalent_carbons {
            if (*c1 == atom1 && *c2 == atom2) || (*c1 == atom2 && *c2 == atom1) {
                return true;
            }
        }
        // Check proton groups
        for group in self.stereo_equivalent_protons {
            if group.contains(&atom1) && group.contains(&atom2) {
                return true;
            }
        }
        false
    }
}

// ============================================================================
// RESIDUE TOPOLOGY DEFINITIONS - All 20 standard amino acids
// ============================================================================

pub static GLYCINE: ResidueTopology = ResidueTopology {
    one_letter: 'G',
    three_letter: "GLY",
    backbone_heavy: &["N", "CA", "C", "O"],
    backbone_protons: &["H", "HA2", "HA3"],  // Two alpha protons
    sidechain_carbons: &[],
    sidechain_protons: &[],
    sidechain_hetero: &[],
    has_backbone_nh: true,
    sidechain_nh_groups: &[],
    stereo_equivalent_carbons: &[],
    stereo_equivalent_protons: &[&["HA2", "HA3"]],
};

pub static ALANINE: ResidueTopology = ResidueTopology {
    one_letter: 'A',
    three_letter: "ALA",
    backbone_heavy: &["N", "CA", "C", "O"],
    backbone_protons: &["H", "HA"],
    sidechain_carbons: &["CB"],
    sidechain_protons: &["HB1", "HB2", "HB3"],  // Methyl
    sidechain_hetero: &[],
    has_backbone_nh: true,
    sidechain_nh_groups: &[],
    stereo_equivalent_carbons: &[],
    stereo_equivalent_protons: &[&["HB1", "HB2", "HB3"]],
};

pub static VALINE: ResidueTopology = ResidueTopology {
    one_letter: 'V',
    three_letter: "VAL",
    backbone_heavy: &["N", "CA", "C", "O"],
    backbone_protons: &["H", "HA"],
    sidechain_carbons: &["CB", "CG1", "CG2"],
    sidechain_protons: &["HB", "HG11", "HG12", "HG13", "HG21", "HG22", "HG23"],
    sidechain_hetero: &[],
    has_backbone_nh: true,
    sidechain_nh_groups: &[],
    // CG1 and CG2 ARE stereo-equivalent in Val (both methyls)
    stereo_equivalent_carbons: &[("CG1", "CG2")],
    stereo_equivalent_protons: &[
        &["HG11", "HG12", "HG13"],  // CG1 methyl
        &["HG21", "HG22", "HG23"],  // CG2 methyl
    ],
};

pub static LEUCINE: ResidueTopology = ResidueTopology {
    one_letter: 'L',
    three_letter: "LEU",
    backbone_heavy: &["N", "CA", "C", "O"],
    backbone_protons: &["H", "HA"],
    sidechain_carbons: &["CB", "CG", "CD1", "CD2"],
    sidechain_protons: &["HB2", "HB3", "HG", "HD11", "HD12", "HD13", "HD21", "HD22", "HD23"],
    sidechain_hetero: &[],
    has_backbone_nh: true,
    sidechain_nh_groups: &[],
    // CD1 and CD2 ARE stereo-equivalent in Leu (both methyls)
    stereo_equivalent_carbons: &[("CD1", "CD2")],
    stereo_equivalent_protons: &[
        &["HB2", "HB3"],
        &["HD11", "HD12", "HD13"],
        &["HD21", "HD22", "HD23"],
    ],
};

pub static ISOLEUCINE: ResidueTopology = ResidueTopology {
    one_letter: 'I',
    three_letter: "ILE",
    backbone_heavy: &["N", "CA", "C", "O"],
    backbone_protons: &["H", "HA"],
    sidechain_carbons: &["CB", "CG1", "CG2", "CD1"],
    sidechain_protons: &["HB", "HG12", "HG13", "HG21", "HG22", "HG23", "HD11", "HD12", "HD13"],
    sidechain_hetero: &[],
    has_backbone_nh: true,
    sidechain_nh_groups: &[],
    // CG1 (~27 ppm) and CG2 (~17 ppm) are NOT stereo-equivalent in Ile!
    // CG1 is methylene, CG2 is methyl - chemically distinct
    stereo_equivalent_carbons: &[],
    stereo_equivalent_protons: &[
        &["HG12", "HG13"],          // CG1 methylene
        &["HG21", "HG22", "HG23"],  // CG2 methyl
        &["HD11", "HD12", "HD13"],  // CD1 methyl
    ],
};

pub static PROLINE: ResidueTopology = ResidueTopology {
    one_letter: 'P',
    three_letter: "PRO",
    backbone_heavy: &["N", "CA", "C", "O"],
    backbone_protons: &["HA"],  // No H - proline has no amide proton!
    sidechain_carbons: &["CB", "CG", "CD"],
    sidechain_protons: &["HB2", "HB3", "HG2", "HG3", "HD2", "HD3"],
    sidechain_hetero: &[],
    has_backbone_nh: false,  // KEY: Proline has NO backbone NH
    sidechain_nh_groups: &[],
    stereo_equivalent_carbons: &[],
    stereo_equivalent_protons: &[
        &["HB2", "HB3"],
        &["HG2", "HG3"],
        &["HD2", "HD3"],
    ],
};

pub static PHENYLALANINE: ResidueTopology = ResidueTopology {
    one_letter: 'F',
    three_letter: "PHE",
    backbone_heavy: &["N", "CA", "C", "O"],
    backbone_protons: &["H", "HA"],
    sidechain_carbons: &["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    sidechain_protons: &["HB2", "HB3", "HD1", "HD2", "HE1", "HE2", "HZ"],
    sidechain_hetero: &[],
    has_backbone_nh: true,
    sidechain_nh_groups: &[],
    // Ring symmetry: CD1/CD2 and CE1/CE2 are equivalent
    stereo_equivalent_carbons: &[("CD1", "CD2"), ("CE1", "CE2")],
    stereo_equivalent_protons: &[
        &["HB2", "HB3"],
        &["HD1", "HD2"],
        &["HE1", "HE2"],
    ],
};

pub static TYROSINE: ResidueTopology = ResidueTopology {
    one_letter: 'Y',
    three_letter: "TYR",
    backbone_heavy: &["N", "CA", "C", "O"],
    backbone_protons: &["H", "HA"],
    sidechain_carbons: &["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    sidechain_protons: &["HB2", "HB3", "HD1", "HD2", "HE1", "HE2", "HH"],
    sidechain_hetero: &["OH"],
    has_backbone_nh: true,
    sidechain_nh_groups: &[],
    stereo_equivalent_carbons: &[("CD1", "CD2"), ("CE1", "CE2")],
    stereo_equivalent_protons: &[
        &["HB2", "HB3"],
        &["HD1", "HD2"],
        &["HE1", "HE2"],
    ],
};

pub static TRYPTOPHAN: ResidueTopology = ResidueTopology {
    one_letter: 'W',
    three_letter: "TRP",
    backbone_heavy: &["N", "CA", "C", "O"],
    backbone_protons: &["H", "HA"],
    sidechain_carbons: &["CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    sidechain_protons: &["HB2", "HB3", "HD1", "HE1", "HE3", "HZ2", "HZ3", "HH2"],
    sidechain_hetero: &["NE1"],
    has_backbone_nh: true,
    // Trp indole NH appears in 15N-HSQC (~129 ppm N, ~10.2 ppm H)
    sidechain_nh_groups: &[SidechainNH::nh("NE1", &["HE1"])],
    stereo_equivalent_carbons: &[],
    stereo_equivalent_protons: &[&["HB2", "HB3"]],
};

pub static SERINE: ResidueTopology = ResidueTopology {
    one_letter: 'S',
    three_letter: "SER",
    backbone_heavy: &["N", "CA", "C", "O"],
    backbone_protons: &["H", "HA"],
    sidechain_carbons: &["CB"],
    sidechain_protons: &["HB2", "HB3", "HG"],
    sidechain_hetero: &["OG"],
    has_backbone_nh: true,
    sidechain_nh_groups: &[],
    stereo_equivalent_carbons: &[],
    stereo_equivalent_protons: &[&["HB2", "HB3"]],
};

pub static THREONINE: ResidueTopology = ResidueTopology {
    one_letter: 'T',
    three_letter: "THR",
    backbone_heavy: &["N", "CA", "C", "O"],
    backbone_protons: &["H", "HA"],
    sidechain_carbons: &["CB", "CG2"],
    sidechain_protons: &["HB", "HG1", "HG21", "HG22", "HG23"],
    sidechain_hetero: &["OG1"],
    has_backbone_nh: true,
    sidechain_nh_groups: &[],
    stereo_equivalent_carbons: &[],
    stereo_equivalent_protons: &[&["HG21", "HG22", "HG23"]],
};

pub static CYSTEINE: ResidueTopology = ResidueTopology {
    one_letter: 'C',
    three_letter: "CYS",
    backbone_heavy: &["N", "CA", "C", "O"],
    backbone_protons: &["H", "HA"],
    sidechain_carbons: &["CB"],
    sidechain_protons: &["HB2", "HB3", "HG"],
    sidechain_hetero: &["SG"],
    has_backbone_nh: true,
    sidechain_nh_groups: &[],
    stereo_equivalent_carbons: &[],
    stereo_equivalent_protons: &[&["HB2", "HB3"]],
};

pub static METHIONINE: ResidueTopology = ResidueTopology {
    one_letter: 'M',
    three_letter: "MET",
    backbone_heavy: &["N", "CA", "C", "O"],
    backbone_protons: &["H", "HA"],
    sidechain_carbons: &["CB", "CG", "CE"],
    sidechain_protons: &["HB2", "HB3", "HG2", "HG3", "HE1", "HE2", "HE3"],
    sidechain_hetero: &["SD"],
    has_backbone_nh: true,
    sidechain_nh_groups: &[],
    stereo_equivalent_carbons: &[],
    stereo_equivalent_protons: &[
        &["HB2", "HB3"],
        &["HG2", "HG3"],
        &["HE1", "HE2", "HE3"],
    ],
};

pub static ASPARAGINE: ResidueTopology = ResidueTopology {
    one_letter: 'N',
    three_letter: "ASN",
    backbone_heavy: &["N", "CA", "C", "O"],
    backbone_protons: &["H", "HA"],
    sidechain_carbons: &["CB", "CG"],  // CG is carbonyl
    sidechain_protons: &["HB2", "HB3", "HD21", "HD22"],
    sidechain_hetero: &["OD1", "ND2"],
    has_backbone_nh: true,
    // Asn sidechain NH2 appears in 15N-HSQC (~113 ppm N, 7.3/6.9 ppm H)
    sidechain_nh_groups: &[SidechainNH::nh2("ND2", &["HD21", "HD22"])],
    stereo_equivalent_carbons: &[],
    stereo_equivalent_protons: &[&["HB2", "HB3"]],
};

pub static GLUTAMINE: ResidueTopology = ResidueTopology {
    one_letter: 'Q',
    three_letter: "GLN",
    backbone_heavy: &["N", "CA", "C", "O"],
    backbone_protons: &["H", "HA"],
    sidechain_carbons: &["CB", "CG", "CD"],  // CD is carbonyl
    sidechain_protons: &["HB2", "HB3", "HG2", "HG3", "HE21", "HE22"],
    sidechain_hetero: &["OE1", "NE2"],
    has_backbone_nh: true,
    // Gln sidechain NH2 appears in 15N-HSQC (~112 ppm N, 7.3/6.9 ppm H)
    sidechain_nh_groups: &[SidechainNH::nh2("NE2", &["HE21", "HE22"])],
    stereo_equivalent_carbons: &[],
    stereo_equivalent_protons: &[
        &["HB2", "HB3"],
        &["HG2", "HG3"],
    ],
};

pub static ASPARTATE: ResidueTopology = ResidueTopology {
    one_letter: 'D',
    three_letter: "ASP",
    backbone_heavy: &["N", "CA", "C", "O"],
    backbone_protons: &["H", "HA"],
    sidechain_carbons: &["CB", "CG"],  // CG is carboxyl
    sidechain_protons: &["HB2", "HB3"],
    sidechain_hetero: &["OD1", "OD2"],
    has_backbone_nh: true,
    sidechain_nh_groups: &[],
    stereo_equivalent_carbons: &[],
    stereo_equivalent_protons: &[&["HB2", "HB3"]],
};

pub static GLUTAMATE: ResidueTopology = ResidueTopology {
    one_letter: 'E',
    three_letter: "GLU",
    backbone_heavy: &["N", "CA", "C", "O"],
    backbone_protons: &["H", "HA"],
    sidechain_carbons: &["CB", "CG", "CD"],  // CD is carboxyl
    sidechain_protons: &["HB2", "HB3", "HG2", "HG3"],
    sidechain_hetero: &["OE1", "OE2"],
    has_backbone_nh: true,
    sidechain_nh_groups: &[],
    stereo_equivalent_carbons: &[],
    stereo_equivalent_protons: &[
        &["HB2", "HB3"],
        &["HG2", "HG3"],
    ],
};

pub static LYSINE: ResidueTopology = ResidueTopology {
    one_letter: 'K',
    three_letter: "LYS",
    backbone_heavy: &["N", "CA", "C", "O"],
    backbone_protons: &["H", "HA"],
    sidechain_carbons: &["CB", "CG", "CD", "CE"],
    sidechain_protons: &["HB2", "HB3", "HG2", "HG3", "HD2", "HD3", "HE2", "HE3", "HZ1", "HZ2", "HZ3"],
    sidechain_hetero: &["NZ"],
    has_backbone_nh: true,
    // Lys NZ (terminal NH3+) rarely visible in 15N-HSQC due to fast exchange
    sidechain_nh_groups: &[],  // Intentionally empty - fast exchange
    stereo_equivalent_carbons: &[],
    stereo_equivalent_protons: &[
        &["HB2", "HB3"],
        &["HG2", "HG3"],
        &["HD2", "HD3"],
        &["HE2", "HE3"],
        &["HZ1", "HZ2", "HZ3"],
    ],
};

pub static ARGININE: ResidueTopology = ResidueTopology {
    one_letter: 'R',
    three_letter: "ARG",
    backbone_heavy: &["N", "CA", "C", "O"],
    backbone_protons: &["H", "HA"],
    sidechain_carbons: &["CB", "CG", "CD", "CZ"],
    sidechain_protons: &["HB2", "HB3", "HG2", "HG3", "HD2", "HD3", "HE", "HH11", "HH12", "HH21", "HH22"],
    sidechain_hetero: &["NE", "NH1", "NH2"],
    has_backbone_nh: true,
    // Arg guanidinium NHs - sometimes visible depending on pH and exchange
    sidechain_nh_groups: &[
        SidechainNH::nh("NE", &["HE"]),
        SidechainNH::nh2("NH1", &["HH11", "HH12"]),
        SidechainNH::nh2("NH2", &["HH21", "HH22"]),
    ],
    stereo_equivalent_carbons: &[],
    stereo_equivalent_protons: &[
        &["HB2", "HB3"],
        &["HG2", "HG3"],
        &["HD2", "HD3"],
        &["HH11", "HH12"],
        &["HH21", "HH22"],
    ],
};

pub static HISTIDINE: ResidueTopology = ResidueTopology {
    one_letter: 'H',
    three_letter: "HIS",
    backbone_heavy: &["N", "CA", "C", "O"],
    backbone_protons: &["H", "HA"],
    sidechain_carbons: &["CB", "CG", "CD2", "CE1"],
    sidechain_protons: &["HB2", "HB3", "HD1", "HD2", "HE1", "HE2"],
    sidechain_hetero: &["ND1", "NE2"],
    has_backbone_nh: true,
    // His ring NHs - tautomer-dependent, sometimes visible
    sidechain_nh_groups: &[
        SidechainNH::nh("ND1", &["HD1"]),
        SidechainNH::nh("NE2", &["HE2"]),
    ],
    stereo_equivalent_carbons: &[],
    stereo_equivalent_protons: &[&["HB2", "HB3"]],
};

// ============================================================================
// LOOKUP FUNCTIONS
// ============================================================================

/// All 20 standard amino acid topologies
pub static ALL_TOPOLOGIES: &[&ResidueTopology] = &[
    &GLYCINE, &ALANINE, &VALINE, &LEUCINE, &ISOLEUCINE,
    &PROLINE, &PHENYLALANINE, &TYROSINE, &TRYPTOPHAN,
    &SERINE, &THREONINE, &CYSTEINE, &METHIONINE,
    &ASPARAGINE, &GLUTAMINE, &ASPARTATE, &GLUTAMATE,
    &LYSINE, &ARGININE, &HISTIDINE,
];

/// Get topology by one-letter code
pub fn get_topology(one_letter: char) -> Option<&'static ResidueTopology> {
    match one_letter.to_ascii_uppercase() {
        'G' => Some(&GLYCINE),
        'A' => Some(&ALANINE),
        'V' => Some(&VALINE),
        'L' => Some(&LEUCINE),
        'I' => Some(&ISOLEUCINE),
        'P' => Some(&PROLINE),
        'F' => Some(&PHENYLALANINE),
        'Y' => Some(&TYROSINE),
        'W' => Some(&TRYPTOPHAN),
        'S' => Some(&SERINE),
        'T' => Some(&THREONINE),
        'C' => Some(&CYSTEINE),
        'M' => Some(&METHIONINE),
        'N' => Some(&ASPARAGINE),
        'Q' => Some(&GLUTAMINE),
        'D' => Some(&ASPARTATE),
        'E' => Some(&GLUTAMATE),
        'K' => Some(&LYSINE),
        'R' => Some(&ARGININE),
        'H' => Some(&HISTIDINE),
        _ => None,
    }
}

/// Get topology by three-letter code
pub fn get_topology_by_three(three_letter: &str) -> Option<&'static ResidueTopology> {
    let upper = three_letter.to_uppercase();
    ALL_TOPOLOGIES.iter().find(|t| t.three_letter == upper).copied()
}

/// Build topologies for a sequence
pub fn sequence_topologies(sequence: &str) -> Vec<&'static ResidueTopology> {
    sequence.chars()
        .filter_map(get_topology)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ile_cg_not_equivalent() {
        // Ile CG1 and CG2 are NOT stereo-equivalent
        assert!(!ISOLEUCINE.are_stereo_equivalent("CG1", "CG2"));
    }

    #[test]
    fn test_val_cg_equivalent() {
        // Val CG1 and CG2 ARE stereo-equivalent
        assert!(VALINE.are_stereo_equivalent("CG1", "CG2"));
    }

    #[test]
    fn test_proline_no_backbone_nh() {
        assert!(!PROLINE.has_backbone_nh);
        assert!(!PROLINE.backbone_protons.contains(&"H"));
    }

    #[test]
    fn test_asn_sidechain_nh2() {
        assert_eq!(ASPARAGINE.sidechain_nh_groups.len(), 1);
        assert_eq!(ASPARAGINE.sidechain_nh_groups[0].nitrogen, "ND2");
        assert!(ASPARAGINE.sidechain_nh_groups[0].is_nh2);
    }

    #[test]
    fn test_trp_indole_nh() {
        assert_eq!(TRYPTOPHAN.sidechain_nh_groups.len(), 1);
        assert_eq!(TRYPTOPHAN.sidechain_nh_groups[0].nitrogen, "NE1");
        assert!(!TRYPTOPHAN.sidechain_nh_groups[0].is_nh2);
    }

    #[test]
    fn test_ile_has_atoms() {
        assert!(ISOLEUCINE.has_atom("CG1"));
        assert!(ISOLEUCINE.has_atom("CG2"));
        assert!(ISOLEUCINE.has_atom("CD1"));
        assert!(!ISOLEUCINE.has_atom("CD2"));  // Ile has NO CD2
        assert!(!ISOLEUCINE.has_atom("CG"));   // Ile has CG1/CG2, not CG
    }

    #[test]
    fn test_gly_alpha_protons() {
        assert!(GLYCINE.backbone_protons.contains(&"HA2"));
        assert!(GLYCINE.backbone_protons.contains(&"HA3"));
        assert!(!GLYCINE.backbone_protons.contains(&"HA"));
    }
}
