//! Tauri commands for chemical shift assignment.

use serde::{Deserialize, Serialize};
use tauri::State;
use uuid::Uuid;

use crate::data::{ChemicalShift, ChemicalShiftList, Molecule, NucleusType};
use crate::testdata::BMRBDatabase;
use crate::error::Result;
use crate::inference::{run_adaptive_assignment, AdaptiveToleranceParams, FactorGraph};
use crate::state::AppState;

/// Result from global assignment.
#[derive(Debug, Serialize)]
pub struct AssignmentResult {
    pub atom_id: String,
    pub atom_name: String,
    pub residue_seq_code: i32,
    pub residue_name: String,
    pub assigned_shift: f64,
    pub confidence: f64,
    pub alternative_shifts: Vec<(f64, f64)>, // (shift, probability)
}

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

/// Parameters for global assignment.
#[derive(Debug, Deserialize)]
pub struct AssignmentParams {
    pub max_iterations: Option<usize>,
    pub tolerance: Option<f64>,
    pub include_sidechain: Option<bool>,
}

/// Run global assignment using factor graph and belief propagation.
///
/// This builds a factor graph from the molecule's atoms, adds BMRB priors,
/// and incorporates observed peak positions from ALL shift lists for this molecule.
/// Returns assignment results with confidence scores and alternatives.
#[tauri::command]
pub async fn run_assignment(
    molecule_id: String,
    _peak_list_id: Option<String>, // Deprecated: now uses all shift lists automatically
    params: Option<AssignmentParams>,
    state: State<'_, AppState>,
) -> Result<Vec<AssignmentResult>> {
    let mol_uuid = Uuid::parse_str(&molecule_id)
        .map_err(|e| crate::error::NmrError::Internal(e.to_string()))?;

    let molecule = state
        .get_molecule(&mol_uuid)
        .ok_or_else(|| crate::error::NmrError::Internal("Molecule not found".to_string()))?;

    // Get parameters or use defaults
    let max_iterations = params.as_ref().and_then(|p| p.max_iterations).unwrap_or(50);
    let tolerance = params.as_ref().and_then(|p| p.tolerance).unwrap_or(1e-6);
    let include_sidechain = params.as_ref().and_then(|p| p.include_sidechain).unwrap_or(true);

    // Load comprehensive BMRB statistics (residue-specific)
    let bmrb_db = BMRBDatabase::load_embedded();

    // Build factor graph
    let mut graph = FactorGraph::new();

    // Process each residue
    let mut atoms_added = 0;
    let mut atoms_no_bmrb = 0;

    for residue in molecule.residues() {
        let atoms = molecule.get_atoms_for_residue(residue.sequence_code);

        for atom in atoms {
            // Skip sidechain atoms if not requested
            let is_backbone = matches!(
                atom.atom_name.as_str(),
                "N" | "CA" | "C" | "O" | "H" | "HA" | "CB"
            );
            if !include_sidechain && !is_backbone {
                continue;
            }

            // Create atom identifier: "residue_seq_code_atom_name"
            let atom_id = format!("{}_{}", residue.sequence_code, atom.atom_name);

            // Get residue-specific BMRB statistics
            let stats = bmrb_db.get(&residue.residue_name, &atom.atom_name);

            if let Some(stats) = stats {
                // Create domain of possible chemical shifts
                // 200 points gives ~0.15 ppm spacing for 30 ppm range
                let domain = generate_shift_domain(stats.min, stats.max, 200);

                // Add variable node
                graph.add_chemical_shift_variable(&atom_id, domain);

                // Add BMRB prior factor with residue-specific mean/std_dev
                graph.add_bmrb_prior_factor(&atom_id, &atom.atom_name, stats.mean, stats.std_dev);
                atoms_added += 1;
            } else {
                // Only warn for backbone atoms that should have BMRB stats
                if matches!(atom.atom_name.as_str(), "N" | "CA" | "C" | "H" | "HA" | "CB") {
                    tracing::warn!("No BMRB stats for {}_{} (residue {})",
                        residue.sequence_code, atom.atom_name, residue.residue_name);
                }
                atoms_no_bmrb += 1;
            }
        }
    }

    tracing::info!("Factor graph: {} atoms added, {} skipped (no BMRB stats)",
        atoms_added, atoms_no_bmrb);

    // Add peak factors from ALL shift lists (not just one)
    // This allows combining data from multiple experiments (HSQC, 13C-HSQC, 1H, etc.)
    {
        let shift_lists = state.shift_lists.read();
        let mut total_shifts_added = 0;
        let mut skipped_no_variable = 0;

        for shift_list in shift_lists.values() {
            // Only use shift lists for this molecule
            if shift_list.molecule_id != mol_uuid {
                continue;
            }

            tracing::info!("Processing shift list '{}' with {} shifts",
                shift_list.name, shift_list.shifts.len());

            for shift in &shift_list.shifts {
                let atom_id = format!("{}_{}", shift.residue_seq_code, shift.atom_name);

                // Check if variable exists (atom must be in the graph)
                if !graph.var_indices().contains_key(&atom_id) {
                    tracing::warn!("Skipping shift for {} (value={:.2}) - no variable in graph",
                        atom_id, shift.value);
                    skipped_no_variable += 1;
                    continue;
                }

                // Compute adaptive tolerance based on domain spacing
                // Tolerance = k * spacing, where k determines sharpness:
                // - k = 1.0 for protons (tighter, smaller range)
                // - k = 1.5 for carbons/nitrogens (standard)
                // This ensures probability concentrates on nearest 1-2 domain points
                let k = if shift.atom_name.starts_with('H') { 1.0 } else { 1.5 };
                let tolerance = graph.get_domain_spacing(&atom_id)
                    .map(|spacing| spacing * k)
                    .unwrap_or(0.5);

                graph.add_peak_factor(
                    vec![atom_id.as_str()],
                    vec![shift.value],
                    vec![tolerance],
                );
                tracing::debug!("Added peak factor for {} = {:.2} ppm", atom_id, shift.value);
                total_shifts_added += 1;
            }
        }

        tracing::info!("Added {} peak factors from {} shift lists (skipped {} without variables)",
            total_shifts_added, shift_lists.len(), skipped_no_variable);
    }

    // Run adaptive belief propagation with entropy-based tolerance optimization
    let adaptive_params = AdaptiveToleranceParams {
        target_normalized_entropy: 0.15, // Target ~15% of max entropy for good confidence
        max_rounds: 5,
        adjustment_factor: 0.7,  // Tighten by 30% each round
        min_tolerance: 0.01,
    };

    let assignments = run_adaptive_assignment(&mut graph, max_iterations, tolerance, adaptive_params);

    // Map to AssignmentResult with residue info
    let results: Vec<AssignmentResult> = assignments
        .into_iter()
        .filter_map(|a| {
            // Parse atom_id to get residue_seq_code and atom_name
            let parts: Vec<&str> = a.atom_id.splitn(2, '_').collect();
            if parts.len() != 2 {
                return None;
            }

            let seq_code: i32 = parts[0].parse().ok()?;
            let atom_name = parts[1].to_string();

            let residue = molecule.get_residue(seq_code)?;

            Some(AssignmentResult {
                atom_id: a.atom_id,
                atom_name,
                residue_seq_code: seq_code,
                residue_name: residue.residue_name.clone(),
                assigned_shift: a.assigned_shift,
                confidence: a.confidence,
                alternative_shifts: a.alternatives,
            })
        })
        .collect();

    Ok(results)
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
    let hnca: Vec<UnlabeledPeak> = vec![];
    let hncacb: Vec<UnlabeledPeak> = vec![];
    let cbcaconh: Vec<UnlabeledPeak> = vec![];
    let hbhaconh: Vec<UnlabeledPeak> = vec![];
    let params = UnifiedAssignmentParams::default();
    let assignments = run_unified_assignment(
        &hsqc_15n, &hsqc_13c, &tocsy, &noesy,
        &hsqc_tocsy_15n, &hsqc_tocsy_13c,
        &hsqc_tocsy_15n_3d, &hsqc_tocsy_13c_3d,
        &hnco, &hnca, &hncacb, &cbcaconh, &hbhaconh,
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

/// Generate a domain of possible chemical shift values.
fn generate_shift_domain(min: f64, max: f64, num_points: usize) -> Vec<f64> {
    let step = (max - min) / (num_points - 1) as f64;
    (0..num_points)
        .map(|i| min + i as f64 * step)
        .collect()
}
