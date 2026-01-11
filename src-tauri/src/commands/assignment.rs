//! Tauri commands for chemical shift assignment.

use serde::{Deserialize, Serialize};
use tauri::State;
use uuid::Uuid;

use crate::data::{ChemicalShift, ChemicalShiftList, Molecule, NucleusType};
use crate::error::Result;
use crate::state::AppState;


/// Load a molecule from sequence.
#[tauri::command]
pub async fn load_molecule_from_sequence(
    name: String,
    sequence: String,
    chain_code: Option<String>,
    state: State<'_, AppState>,
) -> Result<String> {
    let chain = chain_code.unwrap_or_else(|| "A".to_string());
    tracing::info!(
        "Creating molecule '{}' from sequence ({} chars): {}",
        name,
        sequence.len(),
        &sequence
    );
    let molecule = Molecule::from_sequence(&name, &sequence, &chain);
    tracing::info!(
        "Created molecule with {} residues, {} atoms",
        molecule.num_residues(),
        molecule.num_atoms()
    );
    let id = state.add_molecule(molecule);

    state.set_active_molecule(Some(id));

    Ok(id.to_string())
}

/// Get active molecule.
#[tauri::command]
pub async fn get_active_molecule(state: State<'_, AppState>) -> Result<Option<MoleculeInfo>> {
    let molecule = state.get_active_molecule();

    Ok(molecule.map(|m| {
        let num_residues = m.num_residues();
        let num_atoms = m.num_atoms();
        tracing::info!(
            "get_active_molecule: {} - {} residues, {} atoms, sequence len: {}",
            m.name,
            num_residues,
            num_atoms,
            m.sequence.len()
        );
        MoleculeInfo {
            id: m.id.to_string(),
            name: m.name.clone(),
            sequence: m.sequence.clone(),
            num_residues,
            num_atoms,
        }
    }))
}

/// Basic molecule information.
#[derive(Debug, Serialize)]
pub struct MoleculeInfo {
    pub id: String,
    pub name: String,
    pub sequence: String,
    pub num_residues: usize,
    pub num_atoms: usize,
}

/// Get residues for active molecule.
#[tauri::command]
pub async fn get_molecule_residues(
    molecule_id: String,
    state: State<'_, AppState>,
) -> Result<Vec<ResidueInfo>> {
    let uuid = Uuid::parse_str(&molecule_id)
        .map_err(|e| crate::error::NmrError::Internal(e.to_string()))?;

    let molecule = state
        .get_molecule(&uuid)
        .ok_or_else(|| crate::error::NmrError::Internal("Molecule not found".to_string()))?;

    let residues: Vec<ResidueInfo> = molecule
        .residues()
        .map(|r| ResidueInfo {
            id: r.id.to_string(),
            sequence_code: r.sequence_code,
            residue_name: r.residue_name.clone(),
            one_letter_code: r.one_letter_code.to_string(),
            chain_code: r.chain_code.clone(),
        })
        .collect();

    Ok(residues)
}

/// Residue information.
#[derive(Debug, Serialize)]
pub struct ResidueInfo {
    pub id: String,
    pub sequence_code: i32,
    pub residue_name: String,
    pub one_letter_code: String,
    pub chain_code: String,
}

/// Get atoms for a residue.
#[tauri::command]
pub async fn get_residue_atoms(
    molecule_id: String,
    residue_seq_code: i32,
    state: State<'_, AppState>,
) -> Result<Vec<AtomInfo>> {
    let uuid = Uuid::parse_str(&molecule_id)
        .map_err(|e| crate::error::NmrError::Internal(e.to_string()))?;

    let molecule = state
        .get_molecule(&uuid)
        .ok_or_else(|| crate::error::NmrError::Internal("Molecule not found".to_string()))?;

    let atoms: Vec<AtomInfo> = molecule
        .get_atoms_for_residue(residue_seq_code)
        .into_iter()
        .map(|a| AtomInfo {
            id: a.id.to_string(),
            atom_name: a.atom_name.clone(),
            element: format!("{:?}", a.element),
        })
        .collect();

    Ok(atoms)
}

/// Atom information.
#[derive(Debug, Serialize)]
pub struct AtomInfo {
    pub id: String,
    pub atom_name: String,
    pub element: String,
}

/// Create a new chemical shift list.
#[tauri::command]
pub async fn create_shift_list(
    name: String,
    molecule_id: String,
    state: State<'_, AppState>,
) -> Result<String> {
    let mol_uuid = Uuid::parse_str(&molecule_id)
        .map_err(|e| crate::error::NmrError::Internal(e.to_string()))?;

    let list = ChemicalShiftList::new(&name, mol_uuid);
    let id = state.add_shift_list(list);

    Ok(id.to_string())
}

/// Add a chemical shift to a list.
#[tauri::command]
pub async fn add_chemical_shift(
    list_id: String,
    atom_id: String,
    atom_name: String,
    residue_seq_code: i32,
    residue_name: String,
    chain_code: String,
    value: f64,
    nucleus: String,
    state: State<'_, AppState>,
) -> Result<String> {
    let list_uuid = Uuid::parse_str(&list_id)
        .map_err(|e| crate::error::NmrError::Internal(e.to_string()))?;
    let atom_uuid = Uuid::parse_str(&atom_id)
        .map_err(|e| crate::error::NmrError::Internal(e.to_string()))?;

    let nucleus_type = match nucleus.as_str() {
        "H" | "1H" => NucleusType::H1,
        "C" | "13C" => NucleusType::C13,
        "N" | "15N" => NucleusType::N15,
        _ => NucleusType::H1,
    };

    let shift = ChemicalShift::new(
        atom_uuid,
        &atom_name,
        residue_seq_code,
        &residue_name,
        &chain_code,
        value,
        nucleus_type,
    );

    let shift_id = shift.id.to_string();

    // Get the list and add the shift
    let mut lists = state.shift_lists.write();
    if let Some(list) = lists.get_mut(&list_uuid) {
        list.add_shift(shift);
    }

    Ok(shift_id)
}

/// Get chemical shifts for a residue.
#[tauri::command]
pub async fn get_shifts_for_residue(
    list_id: String,
    residue_seq_code: i32,
    state: State<'_, AppState>,
) -> Result<Vec<ShiftInfo>> {
    let list_uuid = Uuid::parse_str(&list_id)
        .map_err(|e| crate::error::NmrError::Internal(e.to_string()))?;

    let list = state
        .get_shift_list(&list_uuid)
        .ok_or_else(|| crate::error::NmrError::Internal("Shift list not found".to_string()))?;

    let shifts: Vec<ShiftInfo> = list
        .get_shifts_for_residue(residue_seq_code)
        .into_iter()
        .map(|s| ShiftInfo {
            id: s.id.to_string(),
            atom_name: s.atom_name.clone(),
            value: s.value,
            error: s.error,
            confidence: s.confidence,
        })
        .collect();

    Ok(shifts)
}

/// Chemical shift information.
#[derive(Debug, Serialize)]
pub struct ShiftInfo {
    pub id: String,
    pub atom_name: String,
    pub value: f64,
    pub error: Option<f64>,
    pub confidence: f64,
}

/// List all chemical shift lists.
#[tauri::command]
pub async fn list_shift_lists(state: State<'_, AppState>) -> Result<Vec<ShiftListInfo>> {
    let lists = state.shift_lists.read();

    Ok(lists
        .values()
        .map(|l| ShiftListInfo {
            id: l.id.to_string(),
            name: l.name.clone(),
            molecule_id: l.molecule_id.to_string(),
            num_shifts: l.shifts.len(),
        })
        .collect())
}

/// Chemical shift list information.
#[derive(Debug, Serialize)]
pub struct ShiftListInfo {
    pub id: String,
    pub name: String,
    pub molecule_id: String,
    pub num_shifts: usize,
}


// ============================================================================
// Real Assignment (from unlabeled peaks)
// ============================================================================

/// Unlabeled peak input for real assignment.
#[derive(Debug, Deserialize)]
pub struct UnlabeledPeakInput {
    pub position_ppm: Vec<f64>,
    pub intensity: f64,
}

/// Result from real assignment.
#[derive(Debug, Serialize)]
pub struct RealAssignmentResult {
    pub spin_system_id: String,
    pub assigned_residue: i32,
    pub residue_name: String,
    pub confidence: f64,
    pub h_shift: f64,
    pub n_shift: f64,
    pub proton_shifts: std::collections::HashMap<String, f64>,
    pub carbon_shifts: std::collections::HashMap<String, f64>,
    pub amino_acid_type: String,
    pub type_confidence: f64,
}

/// Run real assignment from unlabeled peak lists.
///
/// This performs unified NMR assignment using a single factor graph.
/// All evidence (TOCSY, carbon typing, NOESY sequential) is processed simultaneously.
#[tauri::command]
pub async fn run_real_assignment(
    molecule_id: String,
    hsqc_15n_peaks: Vec<UnlabeledPeakInput>,
    hsqc_13c_peaks: Vec<UnlabeledPeakInput>,
    tocsy_peaks: Vec<UnlabeledPeakInput>,
    noesy_peaks: Vec<UnlabeledPeakInput>,
    state: State<'_, AppState>,
) -> Result<Vec<RealAssignmentResult>> {
    use crate::data::{PeakExperimentType, UnlabeledPeak};
    use crate::inference::{run_unified_assignment, UnifiedAssignmentParams, PeakType};

    let mol_uuid = Uuid::parse_str(&molecule_id)
        .map_err(|e| crate::error::NmrError::Internal(e.to_string()))?;

    let molecule = state
        .get_molecule(&mol_uuid)
        .ok_or_else(|| crate::error::NmrError::Internal("Molecule not found".to_string()))?;

    tracing::info!(
        "Starting unified assignment for molecule '{}' with {} residues",
        molecule.name,
        molecule.sequence.len()
    );
    tracing::info!(
        "Input peaks: {} 15N-HSQC, {} 13C-HSQC, {} TOCSY, {} NOESY",
        hsqc_15n_peaks.len(),
        hsqc_13c_peaks.len(),
        tocsy_peaks.len(),
        noesy_peaks.len()
    );

    // Convert input peaks to internal format
    let hsqc_15n: Vec<UnlabeledPeak> = hsqc_15n_peaks
        .into_iter()
        .map(|p| UnlabeledPeak::new(p.position_ppm, p.intensity, PeakExperimentType::Hsqc15N))
        .collect();

    let hsqc_13c: Vec<UnlabeledPeak> = hsqc_13c_peaks
        .into_iter()
        .map(|p| UnlabeledPeak::new(p.position_ppm, p.intensity, PeakExperimentType::Hsqc13C))
        .collect();

    let tocsy: Vec<UnlabeledPeak> = tocsy_peaks
        .into_iter()
        .map(|p| UnlabeledPeak::new(p.position_ppm, p.intensity, PeakExperimentType::Tocsy))
        .collect();

    let noesy: Vec<UnlabeledPeak> = noesy_peaks
        .into_iter()
        .map(|p| UnlabeledPeak::new(p.position_ppm, p.intensity, PeakExperimentType::Noesy))
        .collect();

    // Run unified assignment: ALL evidence in ONE factor graph
    // Note: HSQC-TOCSY and 3D experiments not yet supported in real data loading, pass empty vectors
    let hsqc_tocsy_15n: Vec<UnlabeledPeak> = vec![];
    let hsqc_tocsy_13c: Vec<UnlabeledPeak> = vec![];
    let hsqc_tocsy_15n_3d: Vec<UnlabeledPeak> = vec![];
    let hsqc_tocsy_13c_3d: Vec<UnlabeledPeak> = vec![];
    let hnco: Vec<UnlabeledPeak> = vec![];
    let hncaco: Vec<UnlabeledPeak> = vec![];
    let hnca: Vec<UnlabeledPeak> = vec![];
    let hncacb: Vec<UnlabeledPeak> = vec![];
    let cbcaconh: Vec<UnlabeledPeak> = vec![];
    let hbhaconh: Vec<UnlabeledPeak> = vec![];
    let params = UnifiedAssignmentParams::default();
    let assignments = run_unified_assignment(
        &hsqc_15n, &hsqc_13c, &tocsy, &noesy,
        &hsqc_tocsy_15n, &hsqc_tocsy_13c,
        &hsqc_tocsy_15n_3d, &hsqc_tocsy_13c_3d,
        &hnco, &hncaco, &hnca, &hncacb, &cbcaconh, &hbhaconh,
        &molecule.sequence, &params
    );

    tracing::info!("Unified assignment returned {} results", assignments.len());

    // Convert backbone assignments to results
    let results: Vec<RealAssignmentResult> = assignments
        .into_iter()
        .filter(|a| a.peak_type == PeakType::Backbone && a.assigned_residue > 0)
        .filter_map(|a| {
            // Find the original peak
            let peak = hsqc_15n.iter().find(|p| p.id == a.peak_id)?;
            let residue = molecule.get_residue(a.assigned_residue)?;

            Some(RealAssignmentResult {
                spin_system_id: a.peak_id.to_string(),
                assigned_residue: a.assigned_residue,
                residue_name: residue.residue_name.clone(),
                confidence: a.confidence,
                h_shift: peak.position_ppm[1],
                n_shift: peak.position_ppm[0],
                proton_shifts: std::collections::HashMap::new(),  // Carbon peaks are assigned separately
                carbon_shifts: std::collections::HashMap::new(),
                amino_acid_type: residue.residue_name.clone(),
                type_confidence: a.confidence,
            })
        })
        .collect();

    tracing::info!(
        "Unified assignment complete: {} backbone peaks mapped to {} residues",
        results.len(),
        molecule.sequence.len()
    );

    Ok(results)
}
