//! Structural constraints derived from NMR data.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A distance constraint derived from NOE data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceConstraint {
    pub id: Uuid,
    pub atom1_id: Uuid,
    pub atom2_id: Uuid,
    /// Lower distance bound in Angstroms
    pub lower_bound: f64,
    /// Upper distance bound in Angstroms
    pub upper_bound: f64,
    /// Target distance (center of bounds)
    pub target: f64,
    /// Weight for structure calculation
    pub weight: f64,
    /// Source peak ID if derived from a peak
    pub peak_id: Option<Uuid>,
    /// Constraint type
    pub constraint_type: DistanceConstraintType,
    /// Ambiguity - number of possible atom pairs
    pub ambiguity: u8,
}

/// Type of distance constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceConstraintType {
    /// From NOESY cross-peak
    NOE,
    /// From paramagnetic relaxation enhancement
    PRE,
    /// From hydrogen bond
    HydrogenBond,
    /// From disulfide bridge
    Disulfide,
    /// User-defined
    Manual,
}

impl DistanceConstraint {
    /// Create a new NOE-derived distance constraint.
    pub fn from_noe(
        atom1_id: Uuid,
        atom2_id: Uuid,
        lower: f64,
        upper: f64,
        peak_id: Option<Uuid>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            atom1_id,
            atom2_id,
            lower_bound: lower,
            upper_bound: upper,
            target: (lower + upper) / 2.0,
            weight: 1.0,
            peak_id,
            constraint_type: DistanceConstraintType::NOE,
            ambiguity: 1,
        }
    }

    /// Create standard NOE bounds from intensity class.
    pub fn from_intensity_class(
        atom1_id: Uuid,
        atom2_id: Uuid,
        intensity_class: NOEIntensityClass,
        peak_id: Option<Uuid>,
    ) -> Self {
        let (lower, upper) = intensity_class.to_bounds();
        Self::from_noe(atom1_id, atom2_id, lower, upper, peak_id)
    }

    /// Check if a distance satisfies this constraint.
    pub fn is_satisfied(&self, distance: f64) -> bool {
        distance >= self.lower_bound && distance <= self.upper_bound
    }

    /// Calculate violation (negative if satisfied).
    pub fn violation(&self, distance: f64) -> f64 {
        if distance < self.lower_bound {
            self.lower_bound - distance
        } else if distance > self.upper_bound {
            distance - self.upper_bound
        } else {
            0.0
        }
    }
}

/// NOE intensity classification for distance calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NOEIntensityClass {
    Strong,
    Medium,
    Weak,
    VeryWeak,
}

impl NOEIntensityClass {
    /// Convert intensity class to distance bounds (Angstroms).
    pub fn to_bounds(&self) -> (f64, f64) {
        match self {
            NOEIntensityClass::Strong => (1.8, 2.7),
            NOEIntensityClass::Medium => (1.8, 3.5),
            NOEIntensityClass::Weak => (1.8, 5.0),
            NOEIntensityClass::VeryWeak => (1.8, 6.0),
        }
    }

    /// Classify from NOE volume ratio.
    pub fn from_volume_ratio(ratio: f64) -> Self {
        // Ratio relative to a reference NOE
        if ratio > 0.6 {
            NOEIntensityClass::Strong
        } else if ratio > 0.3 {
            NOEIntensityClass::Medium
        } else if ratio > 0.1 {
            NOEIntensityClass::Weak
        } else {
            NOEIntensityClass::VeryWeak
        }
    }
}

/// A dihedral angle constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DihedralConstraint {
    pub id: Uuid,
    /// Four atoms defining the dihedral
    pub atom_ids: [Uuid; 4],
    /// Target angle in degrees
    pub target_angle: f64,
    /// Angular tolerance in degrees
    pub tolerance: f64,
    /// Weight for structure calculation
    pub weight: f64,
    /// Source of the constraint
    pub source: DihedralSource,
}

/// Source of dihedral constraint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DihedralSource {
    /// From TALOS-N or similar prediction
    ChemicalShiftPrediction(String),
    /// From J-coupling measurement
    JCoupling,
    /// From NOESY pattern
    NOEPattern,
    /// User-defined
    Manual,
}

impl DihedralConstraint {
    pub fn new(atom_ids: [Uuid; 4], target: f64, tolerance: f64, source: DihedralSource) -> Self {
        Self {
            id: Uuid::new_v4(),
            atom_ids,
            target_angle: target,
            tolerance,
            weight: 1.0,
            source,
        }
    }

    /// Check if an angle satisfies this constraint.
    pub fn is_satisfied(&self, angle: f64) -> bool {
        let diff = angle_difference(angle, self.target_angle);
        diff.abs() <= self.tolerance
    }

    /// Calculate angular violation in degrees.
    pub fn violation(&self, angle: f64) -> f64 {
        let diff = angle_difference(angle, self.target_angle);
        let violation = diff.abs() - self.tolerance;
        if violation > 0.0 {
            violation
        } else {
            0.0
        }
    }
}

/// Calculate the smallest difference between two angles in degrees.
fn angle_difference(a: f64, b: f64) -> f64 {
    let mut diff = a - b;
    while diff > 180.0 {
        diff -= 360.0;
    }
    while diff < -180.0 {
        diff += 360.0;
    }
    diff
}

/// A set of constraints for structure calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintSet {
    pub id: Uuid,
    pub name: String,
    pub molecule_id: Uuid,
    pub distance_constraints: Vec<DistanceConstraint>,
    pub dihedral_constraints: Vec<DihedralConstraint>,
}

impl ConstraintSet {
    pub fn new(name: &str, molecule_id: Uuid) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            molecule_id,
            distance_constraints: Vec::new(),
            dihedral_constraints: Vec::new(),
        }
    }

    pub fn add_distance(&mut self, constraint: DistanceConstraint) {
        self.distance_constraints.push(constraint);
    }

    pub fn add_dihedral(&mut self, constraint: DihedralConstraint) {
        self.dihedral_constraints.push(constraint);
    }

    /// Get total number of constraints.
    pub fn total_constraints(&self) -> usize {
        self.distance_constraints.len() + self.dihedral_constraints.len()
    }

    /// Get constraints involving a specific atom.
    pub fn constraints_for_atom(&self, atom_id: &Uuid) -> Vec<&DistanceConstraint> {
        self.distance_constraints
            .iter()
            .filter(|c| &c.atom1_id == atom_id || &c.atom2_id == atom_id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_constraint_satisfaction() {
        let atom1 = Uuid::new_v4();
        let atom2 = Uuid::new_v4();

        let constraint = DistanceConstraint::from_noe(atom1, atom2, 2.0, 4.0, None);

        assert!(constraint.is_satisfied(3.0));
        assert!(!constraint.is_satisfied(1.5));
        assert!(!constraint.is_satisfied(5.0));

        assert_eq!(constraint.violation(3.0), 0.0);
        assert!((constraint.violation(1.5) - 0.5).abs() < 0.001);
        assert!((constraint.violation(5.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_noe_intensity_classification() {
        let bounds = NOEIntensityClass::Strong.to_bounds();
        assert_eq!(bounds, (1.8, 2.7));

        let class = NOEIntensityClass::from_volume_ratio(0.5);
        assert_eq!(class, NOEIntensityClass::Medium);
    }

    #[test]
    fn test_dihedral_constraint() {
        let atoms = [Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4()];
        let constraint = DihedralConstraint::new(
            atoms,
            -60.0,
            30.0,
            DihedralSource::ChemicalShiftPrediction("TALOS-N".to_string()),
        );

        assert!(constraint.is_satisfied(-60.0));
        assert!(constraint.is_satisfied(-45.0));
        assert!(!constraint.is_satisfied(0.0));
    }

    #[test]
    fn test_angle_difference() {
        assert!((angle_difference(10.0, 350.0) - 20.0).abs() < 0.001);
        assert!((angle_difference(350.0, 10.0) - (-20.0)).abs() < 0.001);
        assert!((angle_difference(180.0, -180.0)).abs() < 0.001);
    }
}
