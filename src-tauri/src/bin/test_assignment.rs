//! CLI tool for testing the unified NMR assignment algorithm.
//!
//! Two modes:
//! 1. BMRB: Fetch real chemical shifts from BMRB entries
//! 2. Synthetic: Generate "perfect" peaks from KDE modes
//!
//! Usage:
//!   cargo run --bin test_assignment -- bmrb --entry 4493
//!   cargo run --bin test_assignment -- synthetic --sequence THFG --noise 0.1

use clap::{Parser, Subcommand};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;
use uuid::Uuid;

use nmraster_lib::data::spin_system::UnlabeledPeak;
use nmraster_lib::inference::{
    run_unified_assignment, PeakType, UnifiedAssignmentParams, UnifiedAssignmentResult,
    // New observation-based model
    Observation, NucleusToleranceParams, run_observation_assignment, ObservationAssignmentResult,
};
use nmraster_lib::testdata::{
    extract_sequence, fetch_entry_shifts, filter_residue_range, one_to_three, renumber_shifts,
    DepositedShift, KDEDatabase,
};

#[derive(Parser)]
#[command(name = "test_assignment")]
#[command(about = "Test unified NMR assignment algorithm against ground truth")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    mode: Mode,

    /// Enable verbose output during belief propagation
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Exclude specific experiment types (comma-separated).
    /// Options: 15n-hsqc, 13c-hsqc, tocsy, noesy, hsqc-tocsy-15n, hsqc-tocsy-13c,
    ///          hsqc-tocsy-15n-3d, hsqc-tocsy-13c-3d, hnco, hnca, hncacb, cbcaconh, hbhaconh
    #[arg(long, global = true, conflicts_with = "only")]
    exclude: Option<String>,

    /// Only use specific experiment types (comma-separated).
    /// Options: 15n-hsqc, 13c-hsqc, tocsy, noesy, hsqc-tocsy-15n, hsqc-tocsy-13c,
    ///          hsqc-tocsy-15n-3d, hsqc-tocsy-13c-3d, hnco, hnca, hncacb, cbcaconh, hbhaconh
    #[arg(long, global = true, conflicts_with = "exclude")]
    only: Option<String>,
}

/// Filter for which experiment types to include in the test
#[derive(Clone)]
enum ExperimentFilter {
    All,
    Exclude(HashSet<String>),
    Only(HashSet<String>),
}

impl ExperimentFilter {
    fn from_cli(exclude: &Option<String>, only: &Option<String>) -> Self {
        if let Some(exclude_str) = exclude {
            let excluded: HashSet<String> = exclude_str
                .split(',')
                .map(|s| s.trim().to_lowercase())
                .filter(|s| !s.is_empty())
                .collect();
            ExperimentFilter::Exclude(excluded)
        } else if let Some(only_str) = only {
            let included: HashSet<String> = only_str
                .split(',')
                .map(|s| s.trim().to_lowercase())
                .filter(|s| !s.is_empty())
                .collect();
            ExperimentFilter::Only(included)
        } else {
            ExperimentFilter::All
        }
    }

    fn should_include(&self, exp_type: &str) -> bool {
        match self {
            ExperimentFilter::All => true,
            ExperimentFilter::Exclude(set) => !set.contains(&exp_type.to_lowercase()),
            ExperimentFilter::Only(set) => set.contains(&exp_type.to_lowercase()),
        }
    }
}

#[derive(Subcommand)]
enum Mode {
    /// Fetch real data from a BMRB entry
    Bmrb {
        /// BMRB entry ID (e.g., 4493)
        #[arg(short, long)]
        entry: u32,

        /// Residue range to test (e.g., "45-62"). If not specified, uses all residues.
        #[arg(short, long)]
        residues: Option<String>,

        /// Output JSON results to file
        #[arg(long)]
        output_json: Option<PathBuf>,

        // === Parameter overrides for tuning ===
        /// TOCSY weight at start of BP (default: 1.2)
        #[arg(long)]
        tocsy_init: Option<f64>,

        /// TOCSY weight at end of BP (default: 0.6)
        #[arg(long)]
        tocsy_final: Option<f64>,

        /// Typing weight at start of BP (default: 3.0)
        #[arg(long)]
        typing_init: Option<f64>,

        /// Typing weight at end of BP (default: 6.0)
        #[arg(long)]
        typing_final: Option<f64>,

        /// Sequential NOESY weight (default: 1.0)
        #[arg(long)]
        sequential: Option<f64>,

        /// Sequence type constraint weight (default: 10.0)
        #[arg(long)]
        seq_type: Option<f64>,

        /// Sequence type confidence threshold (default: 0.5)
        #[arg(long)]
        seq_type_thresh: Option<f64>,

        /// Proton tolerance at start (ppm, default: 0.05)
        #[arg(long)]
        h_tol_init: Option<f64>,

        /// Proton tolerance at end (ppm, default: 0.02)
        #[arg(long)]
        h_tol_final: Option<f64>,

        /// Carbon tolerance at start (ppm, default: 3.0)
        #[arg(long)]
        c_tol_init: Option<f64>,

        /// Carbon tolerance at end (ppm, default: 1.0)
        #[arg(long)]
        c_tol_final: Option<f64>,

        /// BP damping factor (default: 0.5)
        #[arg(long)]
        damping: Option<f64>,

        /// Exploration fraction of BP iterations (default: 0.3)
        #[arg(long)]
        explore_frac: Option<f64>,

        /// Maximum BP iterations (default: 100)
        #[arg(long)]
        max_iter: Option<usize>,
    },

    /// Run batch tests across multiple residue windows
    Batch {
        /// BMRB entry ID (e.g., 4493)
        #[arg(short, long)]
        entry: u32,

        /// Output directory for JSON results
        #[arg(short, long)]
        output_dir: PathBuf,

        /// Window sizes to test (comma-separated, e.g., "5,10,20,50")
        #[arg(short, long, default_value = "5,10,20,30,50")]
        windows: String,
    },

    /// Run comprehensive grid search over parameter space
    GridSearch {
        /// BMRB entry ID (e.g., 4493)
        #[arg(short, long)]
        entry: u32,

        /// Residue stretches to test (comma-separated, e.g., "1-10,30-40,60-70")
        #[arg(short, long, default_value = "1-10,30-40,60-70")]
        stretches: String,

        /// Output directory for JSON results
        #[arg(short, long)]
        output_dir: PathBuf,

        /// Grid A index to run (1-9). If not specified, runs all.
        #[arg(long)]
        grid_a: Option<usize>,

        /// Grid B index to run (1-6). If not specified, runs all.
        #[arg(long)]
        grid_b: Option<usize>,

        /// Grid C index to run (1-4). If not specified, runs all.
        #[arg(long)]
        grid_c: Option<usize>,
    },

    /// Generate synthetic data from a sequence
    Synthetic {
        /// One-letter amino acid sequence (e.g., "THFG")
        #[arg(short, long)]
        sequence: String,

        /// Add Gaussian noise (stddev as fraction of typical shift range, 0.0-1.0)
        #[arg(short, long, default_value = "0.0")]
        noise: f64,
    },
}

/// Ground truth tracking for peaks
struct GroundTruth {
    /// Peak ID -> expected residue number (1-indexed)
    peak_to_residue: HashMap<Uuid, i32>,
    /// Peak ID -> atom description (e.g., "H/N", "CA/HA")
    peak_to_atom: HashMap<Uuid, String>,
    /// Peak ID -> observed chemical shifts [(nucleus_name, shift_ppm), ...]
    peak_to_shifts: HashMap<Uuid, Vec<(String, f64)>>,
    /// Peak ID -> reference/expected shifts from KDE [(nucleus_name, expected_ppm), ...]
    peak_to_reference: HashMap<Uuid, Vec<(String, f64)>>,
    /// Expected shifts for each (residue, atom) position from KDE: (residue, atom_name) -> shift_ppm
    expected_atom_shifts: HashMap<(i32, String), f64>,
}

impl GroundTruth {
    fn new() -> Self {
        Self {
            peak_to_residue: HashMap::new(),
            peak_to_atom: HashMap::new(),
            peak_to_shifts: HashMap::new(),
            peak_to_reference: HashMap::new(),
            expected_atom_shifts: HashMap::new(),
        }
    }

    /// Record expected shift for a specific atom position (from KDE mode)
    fn set_expected_shift(&mut self, residue: i32, atom: &str, shift: f64) {
        self.expected_atom_shifts.insert((residue, atom.to_string()), shift);
    }

    fn register(&mut self, peak: &UnlabeledPeak, residue: i32, atom: &str) {
        self.peak_to_residue.insert(peak.id, residue);
        // Normalize stereo-equivalent atoms for fair evaluation
        self.peak_to_atom.insert(peak.id, normalize_stereo_atoms(atom));
        // Store chemical shifts
        let shifts = peak.get_shifts_labeled();
        self.peak_to_shifts.insert(peak.id, shifts);
    }

    /// Register with reference shifts (for synthetic data where we know the KDE modes)
    fn register_with_reference(&mut self, peak: &UnlabeledPeak, residue: i32, atom: &str, reference: Vec<(String, f64)>) {
        self.register(peak, residue, atom);
        self.peak_to_reference.insert(peak.id, reference);
    }
}

/// Normalize stereo-equivalent atom names.
/// In NMR, we cannot distinguish between stereo-equivalent atoms like:
/// - VAL: CG1/CG2, HG1*/HG2* (two equivalent methyl groups)
/// - LEU: CD1/CD2, HD1*/HD2* (two equivalent methyl groups)
/// - PHE/TYR: CD1/CD2, CE1/CE2 (ring symmetry)
/// So treating CG1 as CG2 (same residue) should count as correct.
fn normalize_stereo_atoms(atom_desc: &str) -> String {
    let mut result = atom_desc.to_string();

    // Normalize carbon names: CG1->CG, CG2->CG, CD1->CD, CD2->CD, etc.
    // But keep CA, CB as-is since they're not stereo-equivalent
    let stereo_carbons = [
        ("CG1", "CG"), ("CG2", "CG"),
        ("CD1", "CD"), ("CD2", "CD"),
        ("CE1", "CE"), ("CE2", "CE"), ("CE3", "CE"),
        ("CZ2", "CZ"), ("CZ3", "CZ"),
    ];

    for (from, to) in stereo_carbons {
        result = result.replace(from, to);
    }

    // Normalize proton names attached to stereo-equivalent carbons
    // HG11,HG12,HG13 -> HG1; HG21,HG22,HG23 -> HG2; then HG1,HG2 -> HG
    let stereo_protons = [
        // Methyl protons (3 equivalent H per methyl)
        ("HG11", "HG"), ("HG12", "HG"), ("HG13", "HG"),
        ("HG21", "HG"), ("HG22", "HG"), ("HG23", "HG"),
        ("HD11", "HD"), ("HD12", "HD"), ("HD13", "HD"),
        ("HD21", "HD"), ("HD22", "HD"), ("HD23", "HD"),
        // Methylene protons (2 equivalent H)
        ("HB2", "HB"), ("HB3", "HB"),
        ("HG2", "HG"), ("HG3", "HG"),
        ("HD2", "HD"), ("HD3", "HD"),
        ("HE2", "HE"), ("HE3", "HE"),
        ("HZ2", "HZ"), ("HZ3", "HZ"),
        // Alpha protons for Gly
        ("HA2", "HA"), ("HA3", "HA"),
    ];

    for (from, to) in stereo_protons {
        result = result.replace(from, to);
    }

    result
}

/// Result of comparing assignment to ground truth
struct PeakResult {
    peak_id: Uuid,
    predicted_residue: i32,
    actual_residue: i32,
    confidence: f64,
    is_correct: bool,
    peak_type: PeakType,
    atom_name: String,
    /// Observed chemical shifts for each dimension (e.g., [H, N] or [H, N, C])
    shifts: Vec<(String, f64)>,  // (nucleus_name, shift_ppm)
    /// Reference/expected shifts from KDE (for synthetic data)
    reference_shifts: Vec<(String, f64)>,
}

/// Per amino acid stats
#[derive(Default, Clone)]
struct AAStats {
    total: usize,
    correct: usize,
    backbone_total: usize,
    backbone_correct: usize,
}

/// Summary statistics
struct TestSummary {
    total_peaks: usize,
    correct: usize,
    backbone_peaks: usize,
    backbone_correct: usize,
    results: Vec<PeakResult>,
    per_aa: HashMap<char, AAStats>,
}

/// JSON output structure for automated analysis
#[derive(Serialize)]
struct JsonOutput {
    entry_id: u32,
    residue_range: Option<String>,
    sequence: String,
    num_residues: usize,
    timing_ms: f64,
    overall_accuracy: f64,
    backbone_accuracy: f64,
    peak_counts: PeakCounts,
    per_amino_acid: HashMap<String, JsonAAStats>,
    failures: Vec<JsonFailure>,
}

#[derive(Serialize)]
struct PeakCounts {
    hsqc_15n: usize,
    hsqc_13c: usize,
    tocsy: usize,
    noesy: usize,
    total: usize,
}

#[derive(Serialize)]
struct JsonAAStats {
    total: usize,
    correct: usize,
    accuracy: f64,
}

#[derive(Serialize)]
struct JsonFailure {
    residue: i32,
    amino_acid: char,
    predicted: i32,
    actual: i32,
    atom: String,
    confidence: f64,
}

fn main() {
    let cli = Cli::parse();
    let filter = ExperimentFilter::from_cli(&cli.exclude, &cli.only);

    match cli.mode {
        Mode::Bmrb {
            entry,
            residues,
            output_json,
            tocsy_init,
            tocsy_final,
            typing_init,
            typing_final,
            sequential,
            seq_type,
            seq_type_thresh,
            h_tol_init,
            h_tol_final,
            c_tol_init,
            c_tol_final,
            damping,
            explore_frac,
            max_iter,
        } => {
            // Build params with overrides
            let mut params = UnifiedAssignmentParams {
                verbose: cli.verbose,
                ..Default::default()
            };

            if let Some(v) = tocsy_init { params.tocsy_weight_initial = v; }
            if let Some(v) = tocsy_final { params.tocsy_weight_final = v; }
            if let Some(v) = typing_init { params.typing_weight_initial = v; }
            if let Some(v) = typing_final { params.typing_weight_final = v; }
            if let Some(v) = sequential { params.sequential_weight = v; }
            if let Some(v) = seq_type { params.sequence_type_weight = v; }
            if let Some(v) = seq_type_thresh { params.sequence_type_confidence_threshold = v; }
            if let Some(v) = h_tol_init { params.h_tolerance_initial = v; }
            if let Some(v) = h_tol_final { params.h_tolerance_final = v; }
            if let Some(v) = c_tol_init { params.c_tolerance_initial = v; }
            if let Some(v) = c_tol_final { params.c_tolerance_final = v; }
            if let Some(v) = damping { params.damping = v; }
            if let Some(v) = explore_frac { params.exploration_fraction = v; }
            if let Some(v) = max_iter { params.max_iterations = v; }

            run_bmrb_mode(entry, residues, output_json, params, &filter);
        }
        Mode::Batch { entry, output_dir, windows } => {
            run_batch_mode(entry, output_dir, &windows, cli.verbose, &filter);
        }
        Mode::GridSearch { entry, stretches, output_dir, grid_a, grid_b, grid_c } => {
            run_grid_search_mode(entry, &stretches, output_dir, grid_a, grid_b, grid_c, cli.verbose, &filter);
        }
        Mode::Synthetic { sequence, noise } => {
            run_synthetic_mode(&sequence, noise, cli.verbose, &filter);
        }
    }
}

fn run_bmrb_mode(entry_id: u32, residue_range: Option<String>, output_json: Option<PathBuf>, params: UnifiedAssignmentParams, filter: &ExperimentFilter) {
    println!("Fetching BMRB entry {}...", entry_id);

    // Print filter info
    match filter {
        ExperimentFilter::All => {}
        ExperimentFilter::Exclude(set) => {
            println!("Excluding experiments: {:?}", set);
        }
        ExperimentFilter::Only(set) => {
            println!("Only using experiments: {:?}", set);
        }
    }

    // Print active parameter overrides (if any differ from default)
    let defaults = UnifiedAssignmentParams::default();
    let mut overrides = Vec::new();
    if params.tocsy_weight_initial != defaults.tocsy_weight_initial {
        overrides.push(format!("tocsy_init={:.2}", params.tocsy_weight_initial));
    }
    if params.tocsy_weight_final != defaults.tocsy_weight_final {
        overrides.push(format!("tocsy_final={:.2}", params.tocsy_weight_final));
    }
    if params.typing_weight_initial != defaults.typing_weight_initial {
        overrides.push(format!("typing_init={:.2}", params.typing_weight_initial));
    }
    if params.typing_weight_final != defaults.typing_weight_final {
        overrides.push(format!("typing_final={:.2}", params.typing_weight_final));
    }
    if params.sequential_weight != defaults.sequential_weight {
        overrides.push(format!("sequential={:.2}", params.sequential_weight));
    }
    if params.sequence_type_weight != defaults.sequence_type_weight {
        overrides.push(format!("seq_type={:.2}", params.sequence_type_weight));
    }
    if params.h_tolerance_initial != defaults.h_tolerance_initial {
        overrides.push(format!("h_tol_init={:.3}", params.h_tolerance_initial));
    }
    if params.c_tolerance_initial != defaults.c_tolerance_initial {
        overrides.push(format!("c_tol_init={:.2}", params.c_tolerance_initial));
    }
    if params.damping != defaults.damping {
        overrides.push(format!("damping={:.2}", params.damping));
    }
    if params.exploration_fraction != defaults.exploration_fraction {
        overrides.push(format!("explore={:.2}", params.exploration_fraction));
    }
    if params.max_iterations != defaults.max_iterations {
        overrides.push(format!("max_iter={}", params.max_iterations));
    }
    if !overrides.is_empty() {
        println!("Parameter overrides: {}", overrides.join(", "));
    }

    // Fetch shifts from BMRB
    let mut shifts = match fetch_entry_shifts(entry_id) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error fetching BMRB entry: {}", e);
            std::process::exit(1);
        }
    };

    // Filter to residue range if specified
    if let Some(range) = &residue_range {
        shifts = match filter_residue_range(&shifts, range) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error filtering residues: {}", e);
                std::process::exit(1);
            }
        };
        renumber_shifts(&mut shifts);
    }

    if shifts.is_empty() {
        eprintln!("No shifts found for the specified criteria");
        std::process::exit(1);
    }

    // Extract sequence
    let sequence = extract_sequence(&shifts);
    println!("Sequence: {} ({} residues)", sequence, sequence.len());

    // Print per-residue data density
    print_data_density(&shifts, &sequence);

    // Generate peaks from deposited shifts
    let (hsqc_15n, hsqc_13c, tocsy, noesy, hsqc_tocsy_15n, hsqc_tocsy_13c, hsqc_tocsy_15n_3d, hsqc_tocsy_13c_3d, hnco, hnca, hncacb, cbcaconh, hbhaconh, ground_truth) =
        generate_peaks_from_bmrb(&shifts, filter);

    println!(
        "Generated peaks: {} 15N-HSQC, {} 13C-HSQC, {} TOCSY, {} NOESY, {} 15N-HSQC-TOCSY, {} 13C-HSQC-TOCSY",
        hsqc_15n.len(),
        hsqc_13c.len(),
        tocsy.len(),
        noesy.len(),
        hsqc_tocsy_15n.len(),
        hsqc_tocsy_13c.len()
    );
    println!(
        "3D peaks: {} 15N-HSQC-TOCSY-3D, {} 13C-HSQC-TOCSY-3D, {} HNCO, {} HNCA, {} HNCACB, {} CBCACONH, {} HBHACONH",
        hsqc_tocsy_15n_3d.len(),
        hsqc_tocsy_13c_3d.len(),
        hnco.len(),
        hnca.len(),
        hncacb.len(),
        cbcaconh.len(),
        hbhaconh.len()
    );

    let start = Instant::now();
    let results = run_unified_assignment(
        &hsqc_15n, &hsqc_13c, &tocsy, &noesy, &hsqc_tocsy_15n, &hsqc_tocsy_13c,
        &hsqc_tocsy_15n_3d, &hsqc_tocsy_13c_3d,
        &hnco, &hnca, &hncacb, &cbcaconh, &hbhaconh,
        &sequence, &params
    );
    let elapsed = start.elapsed();
    let timing_ms = elapsed.as_secs_f64() * 1000.0;

    println!("Assignment completed in {:.1}ms", timing_ms);

    // Evaluate and print
    let summary = evaluate_results(&results, &ground_truth, &sequence);
    print_results(&summary, &sequence, &format!("BMRB Entry {}", entry_id));

    // Write JSON output if requested
    if let Some(json_path) = output_json {
        let json_output = build_json_output(
            entry_id,
            residue_range,
            &sequence,
            timing_ms,
            &summary,
            hsqc_15n.len(),
            hsqc_13c.len(),
            tocsy.len(),
            noesy.len(),
        );

        match serde_json::to_string_pretty(&json_output) {
            Ok(json_str) => {
                if let Err(e) = std::fs::write(&json_path, json_str) {
                    eprintln!("Error writing JSON output: {}", e);
                } else {
                    println!("JSON results written to: {}", json_path.display());
                }
            }
            Err(e) => eprintln!("Error serializing JSON: {}", e),
        }
    }
}

fn run_synthetic_mode(sequence: &str, noise: f64, verbose: bool, filter: &ExperimentFilter) {
    println!("Generating synthetic data for sequence: {}", sequence);
    println!("Noise level: {:.1}%", noise * 100.0);

    // Print filter info
    match filter {
        ExperimentFilter::All => {}
        ExperimentFilter::Exclude(set) => {
            println!("Excluding experiments: {:?}", set);
        }
        ExperimentFilter::Only(set) => {
            println!("Only using experiments: {:?}", set);
        }
    }

    // Load KDE database
    let kde = KDEDatabase::load_embedded();

    // Generate peaks from KDE modes
    let (hsqc_15n, hsqc_13c, tocsy, noesy, hsqc_tocsy_15n, hsqc_tocsy_13c, hsqc_tocsy_15n_3d, hsqc_tocsy_13c_3d, hnco, hnca, hncacb, cbcaconh, hbhaconh, ground_truth) =
        generate_synthetic_peaks(sequence, &kde, noise, filter);

    println!(
        "Generated peaks: {} 15N-HSQC, {} 13C-HSQC, {} TOCSY, {} NOESY, {} 15N-HSQC-TOCSY, {} 13C-HSQC-TOCSY",
        hsqc_15n.len(),
        hsqc_13c.len(),
        tocsy.len(),
        noesy.len(),
        hsqc_tocsy_15n.len(),
        hsqc_tocsy_13c.len()
    );
    println!(
        "3D peaks: {} 15N-HSQC-TOCSY-3D, {} 13C-HSQC-TOCSY-3D, {} HNCO, {} HNCA, {} HNCACB, {} CBCACONH, {} HBHACONH",
        hsqc_tocsy_15n_3d.len(),
        hsqc_tocsy_13c_3d.len(),
        hnco.len(),
        hnca.len(),
        hncacb.len(),
        cbcaconh.len(),
        hbhaconh.len()
    );

    // UNIFIED OBSERVATION MODEL: Convert ALL peaks to observations
    let mut observations: Vec<Observation> = Vec::new();

    for peak in &hsqc_15n {
        if let Some(obs) = Observation::from_unlabeled_peak(peak) {
            observations.push(obs);
        }
    }
    for peak in &hsqc_13c {
        if let Some(obs) = Observation::from_unlabeled_peak(peak) {
            observations.push(obs);
        }
    }
    for peak in &tocsy {
        if let Some(obs) = Observation::from_unlabeled_peak(peak) {
            observations.push(obs);
        }
    }
    for peak in &noesy {
        if let Some(obs) = Observation::from_unlabeled_peak(peak) {
            observations.push(obs);
        }
    }
    for peak in &hsqc_tocsy_15n {
        if let Some(obs) = Observation::from_unlabeled_peak(peak) {
            observations.push(obs);
        }
    }
    for peak in &hsqc_tocsy_13c {
        if let Some(obs) = Observation::from_unlabeled_peak(peak) {
            observations.push(obs);
        }
    }
    for peak in &hnca {
        if let Some(obs) = Observation::from_unlabeled_peak(peak) {
            observations.push(obs);
        }
    }
    for peak in &hncacb {
        if let Some(obs) = Observation::from_unlabeled_peak(peak) {
            observations.push(obs);
        }
    }
    for peak in &cbcaconh {
        if let Some(obs) = Observation::from_unlabeled_peak(peak) {
            observations.push(obs);
        }
    }

    println!("Total observations: {}", observations.len());

    let params = UnifiedAssignmentParams {
        verbose,
        ..Default::default()
    };
    let tol_params = NucleusToleranceParams::default();

    let results = run_observation_assignment(&observations, sequence, &params, &tol_params);

    let summary = evaluate_observation_results(&results, &ground_truth, sequence);
    let mode_desc = if noise > 0.0 {
        format!("Synthetic (noise={:.1}%)", noise * 100.0)
    } else {
        "Synthetic (perfect)".to_string()
    };
    print_results(&summary, sequence, &mode_desc);

    // Print the atom-centric chemical shift table
    print_chemical_shift_table(&results, &ground_truth, sequence);
}

/// Generate peaks from BMRB deposited shifts
/// Generates FULL spin system TOCSY, ALL sidechain 13C-HSQC peaks, HSQC-TOCSY (2D/3D), and 3D triple-resonance experiments
fn generate_peaks_from_bmrb(
    shifts: &[DepositedShift],
    filter: &ExperimentFilter,
) -> (
    Vec<UnlabeledPeak>,  // hsqc_15n
    Vec<UnlabeledPeak>,  // hsqc_13c
    Vec<UnlabeledPeak>,  // tocsy
    Vec<UnlabeledPeak>,  // noesy
    Vec<UnlabeledPeak>,  // hsqc_tocsy_15n
    Vec<UnlabeledPeak>,  // hsqc_tocsy_13c
    Vec<UnlabeledPeak>,  // hsqc_tocsy_15n_3d
    Vec<UnlabeledPeak>,  // hsqc_tocsy_13c_3d
    Vec<UnlabeledPeak>,  // hnco
    Vec<UnlabeledPeak>,  // hnca
    Vec<UnlabeledPeak>,  // hncacb
    Vec<UnlabeledPeak>,  // cbcaconh
    Vec<UnlabeledPeak>,  // hbhaconh
    GroundTruth,
) {
    let mut hsqc_15n = Vec::new();
    let mut hsqc_13c = Vec::new();
    let mut tocsy = Vec::new();
    let mut noesy = Vec::new();
    let mut hsqc_tocsy_15n = Vec::new();
    let mut hsqc_tocsy_13c = Vec::new();
    let mut hsqc_tocsy_15n_3d = Vec::new();
    let mut hsqc_tocsy_13c_3d = Vec::new();
    let mut hnco = Vec::new();
    let mut hnca = Vec::new();
    let mut hncacb = Vec::new();
    let mut cbcaconh = Vec::new();
    let mut hbhaconh = Vec::new();
    let mut ground_truth = GroundTruth::new();

    // Group shifts by residue
    let mut by_residue: HashMap<i32, Vec<&DepositedShift>> = HashMap::new();
    for shift in shifts {
        by_residue
            .entry(shift.residue_seq_code)
            .or_default()
            .push(shift);
    }

    // Carbon-proton pairs: carbon name -> possible proton names
    let carbon_proton_pairs: &[(&str, &[&str])] = &[
        ("CA", &["HA", "HA2", "HA3"]),
        ("CB", &["HB", "HB2", "HB3"]),
        ("CG", &["HG", "HG2", "HG3"]),
        ("CG1", &["HG11", "HG12", "HG13"]),
        ("CG2", &["HG21", "HG22", "HG23"]),
        ("CD", &["HD", "HD2", "HD3"]),
        ("CD1", &["HD1", "HD11", "HD12", "HD13"]),
        ("CD2", &["HD2", "HD21", "HD22", "HD23"]),
        ("CE", &["HE", "HE2", "HE3"]),
        ("CE1", &["HE1"]),
        ("CE2", &["HE2"]),
        ("CE3", &["HE3"]),
        ("CZ", &["HZ"]),
        ("CZ2", &["HZ2"]),
        ("CZ3", &["HZ3"]),
        ("CH2", &["HH2"]),
    ];

    // For each residue, generate appropriate peaks
    for (&seq_code, res_shifts) in &by_residue {
        let res_name = &res_shifts[0].residue_name;

        // Find backbone atoms
        let h_shift = res_shifts.iter().find(|s| s.atom_name == "H");
        let n_shift = res_shifts.iter().find(|s| s.atom_name == "N");

        // 15N-HSQC: backbone H-N (skip proline)
        if filter.should_include("15n-hsqc") {
            if let (Some(h), Some(n)) = (h_shift, n_shift) {
                if res_name != "PRO" {
                    let peak = UnlabeledPeak::hsqc_15n(n.shift_value, h.shift_value, 1.0);
                    ground_truth.register(&peak, seq_code, "H/N");
                    hsqc_15n.push(peak);
                }
            }
        }

        // 13C-HSQC: ALL carbon-proton pairs
        if filter.should_include("13c-hsqc") {
            for (carbon_name, proton_names) in carbon_proton_pairs {
                if let Some(c_shift) = res_shifts.iter().find(|s| s.atom_name == *carbon_name) {
                    for proton_name in *proton_names {
                        if let Some(h_shift) = res_shifts.iter().find(|s| s.atom_name == *proton_name) {
                            let intensity = if *carbon_name == "CA" { 1.0 } else { 0.8 };
                            let peak = UnlabeledPeak::hsqc_13c(c_shift.shift_value, h_shift.shift_value, intensity);
                            let atom_desc = format!("{}/{}", carbon_name, proton_name);
                            ground_truth.register(&peak, seq_code, &atom_desc);
                            hsqc_13c.push(peak);
                        }
                    }
                }
            }
        }

        // TOCSY: ALL proton-proton correlations within the residue
        // Collect all proton shifts for this residue
        let proton_shifts: Vec<_> = res_shifts
            .iter()
            .filter(|s| s.atom_name.starts_with('H'))
            .collect();

        if filter.should_include("tocsy") {
            // Generate all pairwise TOCSY correlations
            for i in 0..proton_shifts.len() {
                for j in (i + 1)..proton_shifts.len() {
                    let h1 = proton_shifts[i];
                    let h2 = proton_shifts[j];

                    // Intensity based on how close in the spin system
                    // Backbone H to alpha/beta gets higher intensity
                    let intensity = if h1.atom_name == "H" || h2.atom_name == "H" {
                        1.0
                    } else if h1.atom_name.starts_with("HA") || h2.atom_name.starts_with("HA") {
                        0.9
                    } else {
                        0.7
                    };

                    let peak = UnlabeledPeak::tocsy(h1.shift_value, h2.shift_value, intensity);
                    let atom_desc = format!("{}/{}", h1.atom_name, h2.atom_name);
                    ground_truth.register(&peak, seq_code, &atom_desc);
                    tocsy.push(peak);
                }
            }
        }

        // 15N-HSQC-TOCSY: For each backbone NH, show correlations to all protons in spin system
        // Peak format: (N, H_tocsy) where H_tocsy is any proton in the residue
        if filter.should_include("hsqc-tocsy-15n") {
            if let (Some(h_bb), Some(n)) = (h_shift, n_shift) {
                if res_name != "PRO" {
                    // Generate peaks for all protons in this residue
                    for h_other in &proton_shifts {
                        let intensity = if h_other.atom_name == "H" {
                            1.0  // Backbone H correlation (diagonal-like)
                        } else if h_other.atom_name.starts_with("HA") {
                            0.9  // Alpha protons - strong TOCSY
                        } else if h_other.atom_name.starts_with("HB") {
                            0.8  // Beta protons
                        } else {
                            0.6  // Other sidechain protons
                        };

                        let peak = UnlabeledPeak::hsqc_tocsy_15n(n.shift_value, h_other.shift_value, intensity);
                        let atom_desc = format!("N/{}", h_other.atom_name);
                        ground_truth.register(&peak, seq_code, &atom_desc);
                        hsqc_tocsy_15n.push(peak);
                    }
                }
            }
        }

        // 13C-HSQC-TOCSY: For each C-H pair, show correlations to other protons
        // Peak format: (C, H_tocsy) where H_tocsy is a TOCSY-correlated proton
        if filter.should_include("hsqc-tocsy-13c") {
            for (carbon_name, proton_names) in carbon_proton_pairs {
                if let Some(c_shift) = res_shifts.iter().find(|s| s.atom_name == *carbon_name) {
                    // For this carbon, generate HSQC-TOCSY peaks to all protons in the residue
                    for h_other in &proton_shifts {
                        // Skip the directly attached protons (those are in 13C-HSQC)
                        let is_direct = proton_names.iter().any(|&p| p == h_other.atom_name);

                        let intensity = if is_direct {
                            1.0  // Direct C-H correlation (like HSQC diagonal)
                        } else if h_other.atom_name == "H" {
                            0.7  // Backbone H - usually visible in TOCSY
                        } else {
                            0.6  // Other protons
                        };

                        let peak = UnlabeledPeak::hsqc_tocsy_13c(c_shift.shift_value, h_other.shift_value, intensity);
                        let atom_desc = format!("{}/{}", carbon_name, h_other.atom_name);
                        ground_truth.register(&peak, seq_code, &atom_desc);
                        hsqc_tocsy_13c.push(peak);
                    }
                }
            }
        }

        // 15N-HSQC-TOCSY 3D: (N, H_backbone, H_tocsy)
        // Full 3D version with backbone H dimension for better resolution
        if filter.should_include("hsqc-tocsy-15n-3d") {
            if let (Some(h_bb), Some(n)) = (h_shift, n_shift) {
                if res_name != "PRO" {
                    for h_other in &proton_shifts {
                        let intensity = if h_other.atom_name == "H" {
                            1.0
                        } else if h_other.atom_name.starts_with("HA") {
                            0.9
                        } else if h_other.atom_name.starts_with("HB") {
                            0.8
                        } else {
                            0.6
                        };

                        let peak = UnlabeledPeak::hsqc_tocsy_15n_3d(n.shift_value, h_bb.shift_value, h_other.shift_value, intensity);
                        let atom_desc = format!("N/H/{}", h_other.atom_name);
                        ground_truth.register(&peak, seq_code, &atom_desc);
                        hsqc_tocsy_15n_3d.push(peak);
                    }
                }
            }
        }

        // 13C-HSQC-TOCSY 3D: (C, H_anchor, H_tocsy)
        // Full 3D version with anchor H dimension for better resolution
        if filter.should_include("hsqc-tocsy-13c-3d") {
            for (carbon_name, proton_names) in carbon_proton_pairs {
                if let Some(c_shift) = res_shifts.iter().find(|s| s.atom_name == *carbon_name) {
                    // Find the attached proton(s) to use as anchor
                    let attached_protons: Vec<_> = proton_names.iter()
                        .filter_map(|&p| res_shifts.iter().find(|s| s.atom_name == p))
                        .collect();

                    // Use first attached proton as anchor
                    if let Some(h_anchor) = attached_protons.first() {
                        for h_other in &proton_shifts {
                            let is_anchor = h_other.atom_name == h_anchor.atom_name;
                            let intensity = if is_anchor {
                                1.0  // Anchor proton (diagonal-like)
                            } else if h_other.atom_name == "H" {
                                0.7
                            } else {
                                0.6
                            };

                            let peak = UnlabeledPeak::hsqc_tocsy_13c_3d(
                                c_shift.shift_value,
                                h_anchor.shift_value,
                                h_other.shift_value,
                                intensity
                            );
                            let atom_desc = format!("{}/{}/{}", carbon_name, h_anchor.atom_name, h_other.atom_name);
                            ground_truth.register(&peak, seq_code, &atom_desc);
                            hsqc_tocsy_13c_3d.push(peak);
                        }
                    }
                }
            }
        }
    }

    // Sequential NOESY: H(i) to HA(i-1)
    let mut seq_codes: Vec<_> = by_residue.keys().copied().collect();
    seq_codes.sort();

    if filter.should_include("noesy") {
        for i in 1..seq_codes.len() {
            let curr_seq = seq_codes[i];
            let prev_seq = seq_codes[i - 1];

            if curr_seq != prev_seq + 1 {
                continue; // Skip non-sequential
            }

            let curr_shifts = &by_residue[&curr_seq];
            let prev_shifts = &by_residue[&prev_seq];

            let h_curr = curr_shifts.iter().find(|s| s.atom_name == "H");
            let ha_prev = prev_shifts
                .iter()
                .find(|s| s.atom_name == "HA" || s.atom_name == "HA2");

            if let (Some(h), Some(ha)) = (h_curr, ha_prev) {
                let peak = UnlabeledPeak::noesy(h.shift_value, ha.shift_value, 1.0);
                // Register with current residue (where backbone H is)
                let atom_desc = format!("H(i)/HA(i-1)");
                ground_truth.register(&peak, curr_seq, &atom_desc);
                noesy.push(peak);
            }
        }
    }

    // 3D Triple-Resonance Experiments
    // These provide unambiguous sequential connectivity via CA/CB chemical shift matching
    for i in 1..seq_codes.len() {
        let curr_seq = seq_codes[i];
        let prev_seq = seq_codes[i - 1];

        if curr_seq != prev_seq + 1 {
            continue; // Skip non-sequential
        }

        let curr_shifts = &by_residue[&curr_seq];
        let prev_shifts = &by_residue[&prev_seq];
        let curr_name = &curr_shifts[0].residue_name;

        // Skip proline at current position (no backbone NH)
        if curr_name == "PRO" {
            continue;
        }

        // Get backbone H and N for current residue
        let h_curr = curr_shifts.iter().find(|s| s.atom_name == "H");
        let n_curr = curr_shifts.iter().find(|s| s.atom_name == "N");

        let (h_val, n_val) = match (h_curr, n_curr) {
            (Some(h), Some(n)) => (h.shift_value, n.shift_value),
            _ => continue,
        };

        // Get carbon/proton shifts for current and previous residues
        let ca_curr = curr_shifts.iter().find(|s| s.atom_name == "CA").map(|s| s.shift_value);
        let cb_curr = curr_shifts.iter().find(|s| s.atom_name == "CB").map(|s| s.shift_value);
        let ca_prev = prev_shifts.iter().find(|s| s.atom_name == "CA").map(|s| s.shift_value);
        let cb_prev = prev_shifts.iter().find(|s| s.atom_name == "CB").map(|s| s.shift_value);
        let co_prev = prev_shifts.iter().find(|s| s.atom_name == "C").map(|s| s.shift_value);
        let ha_prev = prev_shifts.iter().find(|s| s.atom_name == "HA" || s.atom_name == "HA2").map(|s| s.shift_value);
        let hb_prev = prev_shifts.iter().find(|s| s.atom_name == "HB" || s.atom_name == "HB2").map(|s| s.shift_value);

        // HNCO: (H, N, CO) - correlates NH(i) with CO(i-1)
        // Ground truth: the CO belongs to residue i-1
        if filter.should_include("hnco") {
            if let Some(co) = co_prev {
                let peak = UnlabeledPeak::hnco(h_val, n_val, co, 1.0);
                ground_truth.register(&peak, curr_seq - 1, "H/N/C(i-1)");
                hnco.push(peak);
            }
        }

        // HNCA: (H, N, CA) - correlates NH(i) with CA(i) strong, CA(i-1) weak
        if filter.should_include("hnca") {
            // Intra-residue CA(i) - strong
            if let Some(ca) = ca_curr {
                let peak = UnlabeledPeak::hnca(h_val, n_val, ca, 1.0);
                let reference = vec![
                    ("H".to_string(), h_val),
                    ("N".to_string(), n_val),
                    ("CA".to_string(), ca),
                ];
                ground_truth.register_with_reference(&peak, curr_seq, "H/N/CA(i)", reference);
                hnca.push(peak);
            }
            // Inter-residue CA(i-1) - weak (~30%)
            if let Some(ca) = ca_prev {
                let peak = UnlabeledPeak::hnca(h_val, n_val, ca, 0.3);
                // Ground truth: the carbon BELONGS to residue i-1
                let reference = vec![
                    ("H".to_string(), h_val),
                    ("N".to_string(), n_val),
                    ("CA".to_string(), ca),
                ];
                ground_truth.register_with_reference(&peak, curr_seq - 1, "H/N/CA(i-1)", reference);
                hnca.push(peak);
            }
        }

        // HNCACB: (H, N, C) - positive for intra (i), negative for inter (i-1)
        if filter.should_include("hncacb") {
            // Intra-residue CA(i) - positive
            if let Some(ca) = ca_curr {
                let peak = UnlabeledPeak::hncacb(h_val, n_val, ca, 1.0);
                let reference = vec![
                    ("H".to_string(), h_val),
                    ("N".to_string(), n_val),
                    ("CA".to_string(), ca),
                ];
                ground_truth.register_with_reference(&peak, curr_seq, "H/N/CA(i)+", reference);
                hncacb.push(peak);
            }
            // Intra-residue CB(i) - positive
            if let Some(cb) = cb_curr {
                let peak = UnlabeledPeak::hncacb(h_val, n_val, cb, 0.8);
                let reference = vec![
                    ("H".to_string(), h_val),
                    ("N".to_string(), n_val),
                    ("CB".to_string(), cb),
                ];
                ground_truth.register_with_reference(&peak, curr_seq, "H/N/CB(i)+", reference);
                hncacb.push(peak);
            }
            // Inter-residue CA(i-1) - negative
            // Ground truth: the carbon BELONGS to residue i-1, not current backbone
            if let Some(ca) = ca_prev {
                let peak = UnlabeledPeak::hncacb(h_val, n_val, ca, -0.3);
                let reference = vec![
                    ("H".to_string(), h_val),
                    ("N".to_string(), n_val),
                    ("CA".to_string(), ca),
                ];
                ground_truth.register_with_reference(&peak, curr_seq - 1, "H/N/CA(i-1)-", reference);
                hncacb.push(peak);
            }
            // Inter-residue CB(i-1) - negative
            if let Some(cb) = cb_prev {
                let peak = UnlabeledPeak::hncacb(h_val, n_val, cb, -0.25);
                let reference = vec![
                    ("H".to_string(), h_val),
                    ("N".to_string(), n_val),
                    ("CB".to_string(), cb),
                ];
                ground_truth.register_with_reference(&peak, curr_seq - 1, "H/N/CB(i-1)-", reference);
                hncacb.push(peak);
            }
        }

        // CBCACONH: (H, N, CA/CB) - only i-1 carbons
        // Ground truth: ALL peaks show previous residue's carbons
        if filter.should_include("cbcaconh") {
            if let Some(ca) = ca_prev {
                let peak = UnlabeledPeak::cbcaconh(h_val, n_val, ca, 1.0);
                ground_truth.register(&peak, curr_seq - 1, "H/N/CA(i-1)");
                cbcaconh.push(peak);
            }
            if let Some(cb) = cb_prev {
                let peak = UnlabeledPeak::cbcaconh(h_val, n_val, cb, 0.8);
                ground_truth.register(&peak, curr_seq - 1, "H/N/CB(i-1)");
                cbcaconh.push(peak);
            }
        }

        // HBHACONH: (H, N, HA/HB) - only i-1 protons
        // Ground truth: ALL peaks show previous residue's protons
        if filter.should_include("hbhaconh") {
            if let Some(ha) = ha_prev {
                let peak = UnlabeledPeak::hbhaconh(h_val, n_val, ha, 1.0);
                ground_truth.register(&peak, curr_seq - 1, "H/N/HA(i-1)");
                hbhaconh.push(peak);
            }
            if let Some(hb) = hb_prev {
                let peak = UnlabeledPeak::hbhaconh(h_val, n_val, hb, 0.8);
                ground_truth.register(&peak, curr_seq - 1, "H/N/HB(i-1)");
                hbhaconh.push(peak);
            }
        }
    }

    (hsqc_15n, hsqc_13c, tocsy, noesy, hsqc_tocsy_15n, hsqc_tocsy_13c, hsqc_tocsy_15n_3d, hsqc_tocsy_13c_3d, hnco, hnca, hncacb, cbcaconh, hbhaconh, ground_truth)
}

/// Generate synthetic peaks from KDE modes
/// Generates FULL spin system TOCSY, ALL sidechain 13C-HSQC peaks, HSQC-TOCSY, and 3D triple-resonance experiments
fn generate_synthetic_peaks(
    sequence: &str,
    kde: &KDEDatabase,
    noise: f64,
    filter: &ExperimentFilter,
) -> (
    Vec<UnlabeledPeak>,  // hsqc_15n
    Vec<UnlabeledPeak>,  // hsqc_13c
    Vec<UnlabeledPeak>,  // tocsy
    Vec<UnlabeledPeak>,  // noesy
    Vec<UnlabeledPeak>,  // hsqc_tocsy_15n
    Vec<UnlabeledPeak>,  // hsqc_tocsy_13c
    Vec<UnlabeledPeak>,  // hsqc_tocsy_15n_3d
    Vec<UnlabeledPeak>,  // hsqc_tocsy_13c_3d
    Vec<UnlabeledPeak>,  // hnco
    Vec<UnlabeledPeak>,  // hnca
    Vec<UnlabeledPeak>,  // hncacb
    Vec<UnlabeledPeak>,  // cbcaconh
    Vec<UnlabeledPeak>,  // hbhaconh
    GroundTruth,
) {
    use rand::Rng;

    let mut rng = rand::rng();
    let mut hsqc_15n = Vec::new();
    let mut hsqc_13c = Vec::new();
    let mut tocsy = Vec::new();
    let mut noesy = Vec::new();
    let mut hsqc_tocsy_15n = Vec::new();
    let mut hsqc_tocsy_13c = Vec::new();
    let mut hsqc_tocsy_15n_3d = Vec::new();
    let mut hsqc_tocsy_13c_3d = Vec::new();
    let mut hnco = Vec::new();
    let mut hnca = Vec::new();
    let mut hncacb = Vec::new();
    let mut cbcaconh = Vec::new();
    let mut hbhaconh = Vec::new();
    let mut ground_truth = GroundTruth::new();

    // Carbon-proton pairs for 13C-HSQC (same as BMRB mode)
    let carbon_proton_pairs: &[(&str, &[&str])] = &[
        ("CA", &["HA", "HA2", "HA3"]),
        ("CB", &["HB", "HB2", "HB3"]),
        ("CG", &["HG", "HG2", "HG3"]),
        ("CG1", &["HG11", "HG12", "HG13"]),
        ("CG2", &["HG21", "HG22", "HG23"]),
        ("CD", &["HD", "HD2", "HD3"]),
        ("CD1", &["HD1", "HD11", "HD12", "HD13"]),
        ("CD2", &["HD2", "HD21", "HD22", "HD23"]),
        ("CE", &["HE", "HE2", "HE3"]),
        ("CE1", &["HE1"]),
        ("CE2", &["HE2"]),
        ("CE3", &["HE3"]),
        ("CZ", &["HZ"]),
        ("CZ2", &["HZ2"]),
        ("CZ3", &["HZ3"]),
        ("CH2", &["HH2"]),
    ];

    // All proton atom names to check for TOCSY
    let proton_atoms = &[
        "H", "HA", "HA2", "HA3",
        "HB", "HB2", "HB3",
        "HG", "HG2", "HG3", "HG11", "HG12", "HG13", "HG21", "HG22", "HG23",
        "HD", "HD1", "HD2", "HD3", "HD11", "HD12", "HD13", "HD21", "HD22", "HD23",
        "HE", "HE1", "HE2", "HE3",
        "HZ", "HZ2", "HZ3",
        "HH", "HH2", "HH11", "HH12", "HH21", "HH22",
    ];

    // Store shifts for sequential experiments (NOESY, 3D triple-resonance)
    // Store observed shifts (potentially noisy) for sequential experiments
    let mut h_shifts: HashMap<i32, f64> = HashMap::new();
    let mut ha_shifts: HashMap<i32, f64> = HashMap::new();
    let mut hb_shifts: HashMap<i32, f64> = HashMap::new();
    let mut n_shifts: HashMap<i32, f64> = HashMap::new();
    let mut ca_shifts: HashMap<i32, f64> = HashMap::new();
    let mut cb_shifts: HashMap<i32, f64> = HashMap::new();
    let mut co_shifts: HashMap<i32, f64> = HashMap::new();
    // Store reference shifts (KDE modes) for sequential experiments
    let mut h_refs: HashMap<i32, f64> = HashMap::new();
    let mut ha_refs: HashMap<i32, f64> = HashMap::new();
    let mut hb_refs: HashMap<i32, f64> = HashMap::new();
    let mut n_refs: HashMap<i32, f64> = HashMap::new();
    let mut ca_refs: HashMap<i32, f64> = HashMap::new();
    let mut cb_refs: HashMap<i32, f64> = HashMap::new();
    let mut co_refs: HashMap<i32, f64> = HashMap::new();

    // Add noise helper
    let add_noise = |value: f64, typical_std: f64, rng: &mut rand::rngs::ThreadRng| -> f64 {
        if noise > 0.0 {
            let noise_std = noise * typical_std;
            let z: f64 = rng.random_range(-3.0..3.0);
            value + z * noise_std
        } else {
            value
        }
    };

    for (i, one_letter) in sequence.chars().enumerate() {
        let seq_code = (i + 1) as i32;
        let res_name = one_to_three(one_letter);

        // Collect all proton shifts for this residue (for TOCSY)
        // Format: (atom_name, observed_shift, reference_shift)
        let mut residue_protons: Vec<(String, f64, f64)> = Vec::new();
        // Collect carbon shifts for HSQC-TOCSY-13C
        // Format: (carbon_name, observed_shift, reference_shift, direct_protons)
        let mut residue_carbons: Vec<(String, f64, f64, Vec<&str>)> = Vec::new();

        // Get backbone H and N
        let h_mode = kde.get(res_name, "H").map(|g| g.mode());
        let n_mode = kde.get(res_name, "N").map(|g| g.mode());

        // 15N-HSQC: backbone H-N (skip proline, skip position 1 for N-terminus)
        if one_letter != 'P' && i > 0 {
            if let (Some(h), Some(n)) = (h_mode, n_mode) {
                // Record expected shifts for atom-centric table
                ground_truth.set_expected_shift(seq_code, "H", h);
                ground_truth.set_expected_shift(seq_code, "N", n);

                let h_shift = add_noise(h, 0.5, &mut rng);
                let n_shift = add_noise(n, 4.0, &mut rng);

                if filter.should_include("15n-hsqc") {
                    let peak = UnlabeledPeak::hsqc_15n(n_shift, h_shift, 1.0);
                    let reference = vec![
                        ("N".to_string(), n),  // KDE mode
                        ("H".to_string(), h),  // KDE mode
                    ];
                    ground_truth.register_with_reference(&peak, seq_code, "H/N", reference);
                    hsqc_15n.push(peak);
                }

                h_shifts.insert(seq_code, h_shift);
                n_shifts.insert(seq_code, n_shift);
                h_refs.insert(seq_code, h);  // KDE mode reference
                n_refs.insert(seq_code, n);  // KDE mode reference
                residue_protons.push(("H".to_string(), h_shift, h));  // (name, observed, reference)
            }
        }

        // 13C-HSQC: ALL carbon-proton pairs from KDE
        for (carbon_name, proton_names) in carbon_proton_pairs {
            if let Some(c_kde) = kde.get(res_name, carbon_name) {
                let c_mode = c_kde.mode();  // KDE mode (reference)
                // Record expected carbon shift for atom-centric table
                ground_truth.set_expected_shift(seq_code, carbon_name, c_mode);

                let c_shift = add_noise(c_mode, 2.0, &mut rng);
                residue_carbons.push((carbon_name.to_string(), c_shift, c_mode, proton_names.to_vec()));

                // Track CA and CB for 3D triple-resonance experiments
                if *carbon_name == "CA" {
                    ca_shifts.insert(seq_code, c_shift);
                    ca_refs.insert(seq_code, c_mode);  // KDE mode reference
                } else if *carbon_name == "CB" {
                    cb_shifts.insert(seq_code, c_shift);
                    cb_refs.insert(seq_code, c_mode);  // KDE mode reference
                }

                for proton_name in *proton_names {
                    if let Some(h_kde) = kde.get(res_name, proton_name) {
                        let h_mode = h_kde.mode();  // KDE mode (reference)
                        // Record expected proton shift for atom-centric table
                        ground_truth.set_expected_shift(seq_code, proton_name, h_mode);
                        let h_shift = add_noise(h_mode, 0.3, &mut rng);

                        if filter.should_include("13c-hsqc") {
                            let intensity = if *carbon_name == "CA" { 1.0 } else { 0.8 };
                            let peak = UnlabeledPeak::hsqc_13c(c_shift, h_shift, intensity);
                            let atom_desc = format!("{}/{}", carbon_name, proton_name);
                            let reference = vec![
                                ("C".to_string(), c_mode),  // KDE mode
                                ("H".to_string(), h_mode),  // KDE mode
                            ];
                            ground_truth.register_with_reference(&peak, seq_code, &atom_desc, reference);
                            hsqc_13c.push(peak);
                        }

                        // Track HA/HB for sequential experiments
                        if *proton_name == "HA" || *proton_name == "HA2" {
                            ha_shifts.insert(seq_code, h_shift);
                            ha_refs.insert(seq_code, h_mode);  // KDE mode reference
                        }
                        if *proton_name == "HB" || *proton_name == "HB2" {
                            hb_shifts.insert(seq_code, h_shift);
                            hb_refs.insert(seq_code, h_mode);  // KDE mode reference
                        }

                        // Add to proton list for TOCSY (avoid duplicates)
                        if !residue_protons.iter().any(|(name, _, _)| name == *proton_name) {
                            residue_protons.push((proton_name.to_string(), h_shift, h_mode));  // (name, observed, reference)
                        }
                    }
                }
            }
        }

        // Track CO (carbonyl carbon) for HNCO - use "C" atom name
        if let Some(co_kde) = kde.get(res_name, "C") {
            let co_mode = co_kde.mode();  // KDE mode (reference)
            let co_shift = add_noise(co_mode, 2.0, &mut rng);
            co_shifts.insert(seq_code, co_shift);
            co_refs.insert(seq_code, co_mode);  // KDE mode reference
        }

        // Also collect any protons not attached to carbons (for completeness)
        for proton_name in proton_atoms {
            if !residue_protons.iter().any(|(name, _, _)| name == *proton_name) {
                if let Some(h_kde) = kde.get(res_name, proton_name) {
                    let h_mode = h_kde.mode();  // KDE mode (reference)
                    let h_shift = add_noise(h_mode, 0.3, &mut rng);
                    residue_protons.push((proton_name.to_string(), h_shift, h_mode));

                    // Track HA for NOESY
                    if *proton_name == "HA" || *proton_name == "HA2" {
                        ha_shifts.insert(seq_code, h_shift);
                    }
                }
            }
        }

        // TOCSY: ALL proton-proton correlations within the residue
        if filter.should_include("tocsy") {
            for i in 0..residue_protons.len() {
                for j in (i + 1)..residue_protons.len() {
                    let (name1, shift1, ref1) = &residue_protons[i];
                    let (name2, shift2, ref2) = &residue_protons[j];

                    // Intensity based on how close in the spin system
                    let intensity = if name1 == "H" || name2 == "H" {
                        1.0
                    } else if name1.starts_with("HA") || name2.starts_with("HA") {
                        0.9
                    } else {
                        0.7
                    };

                    let peak = UnlabeledPeak::tocsy(*shift1, *shift2, intensity);
                    let atom_desc = format!("{}/{}", name1, name2);
                    let reference = vec![
                        ("H1".to_string(), *ref1),  // KDE mode for first proton
                        ("H2".to_string(), *ref2),  // KDE mode for second proton
                    ];
                    ground_truth.register_with_reference(&peak, seq_code, &atom_desc, reference);
                    tocsy.push(peak);
                }
            }
        }

        // 15N-HSQC-TOCSY: For each backbone NH, show correlations to all protons in spin system
        if filter.should_include("hsqc-tocsy-15n") && one_letter != 'P' && i > 0 {
            if let (Some(&n_shift), Some(n_ref)) = (n_shifts.get(&seq_code), n_mode) {
                for (h_name, h_shift, h_ref) in &residue_protons {
                    let intensity = if h_name == "H" {
                        1.0  // Backbone H correlation (diagonal-like)
                    } else if h_name.starts_with("HA") {
                        0.9  // Alpha protons - strong TOCSY
                    } else if h_name.starts_with("HB") {
                        0.8  // Beta protons
                    } else {
                        0.6  // Other sidechain protons
                    };

                    let peak = UnlabeledPeak::hsqc_tocsy_15n(n_shift, *h_shift, intensity);
                    let atom_desc = format!("N/{}", h_name);
                    let reference = vec![
                        ("N".to_string(), n_ref),   // KDE mode for N
                        ("H".to_string(), *h_ref),  // KDE mode for H
                    ];
                    ground_truth.register_with_reference(&peak, seq_code, &atom_desc, reference);
                    hsqc_tocsy_15n.push(peak);
                }
            }
        }

        // 13C-HSQC-TOCSY: For each C-H pair, show correlations to other protons
        if filter.should_include("hsqc-tocsy-13c") {
            for (carbon_name, c_shift, c_ref, direct_protons) in &residue_carbons {
                for (h_name, h_shift, h_ref) in &residue_protons {
                    let is_direct = direct_protons.iter().any(|&p| p == h_name);

                    let intensity = if is_direct {
                        1.0  // Direct C-H correlation (like HSQC diagonal)
                    } else if h_name == "H" {
                        0.7  // Backbone H - usually visible in TOCSY
                    } else {
                        0.6  // Other protons
                    };

                    let peak = UnlabeledPeak::hsqc_tocsy_13c(*c_shift, *h_shift, intensity);
                    let atom_desc = format!("{}/{}", carbon_name, h_name);
                    let reference = vec![
                        ("C".to_string(), *c_ref),   // KDE mode for C
                        ("H".to_string(), *h_ref),   // KDE mode for H
                    ];
                    ground_truth.register_with_reference(&peak, seq_code, &atom_desc, reference);
                    hsqc_tocsy_13c.push(peak);
                }
            }
        }

        // 15N-HSQC-TOCSY 3D: (N, H_backbone, H_tocsy)
        if filter.should_include("hsqc-tocsy-15n-3d") && one_letter != 'P' && i > 0 {
            if let (Some(&n_shift), Some(&h_bb), Some(n_ref), Some(h_bb_ref)) =
                (n_shifts.get(&seq_code), h_shifts.get(&seq_code), n_mode, h_mode) {
                for (h_name, h_shift, h_ref) in &residue_protons {
                    let intensity = if h_name == "H" {
                        1.0
                    } else if h_name.starts_with("HA") {
                        0.9
                    } else if h_name.starts_with("HB") {
                        0.8
                    } else {
                        0.6
                    };

                    let peak = UnlabeledPeak::hsqc_tocsy_15n_3d(n_shift, h_bb, *h_shift, intensity);
                    let atom_desc = format!("N/H/{}", h_name);
                    let reference = vec![
                        ("N".to_string(), n_ref),     // KDE mode for N
                        ("H_bb".to_string(), h_bb_ref), // KDE mode for backbone H
                        ("H".to_string(), *h_ref),    // KDE mode for TOCSY H
                    ];
                    ground_truth.register_with_reference(&peak, seq_code, &atom_desc, reference);
                    hsqc_tocsy_15n_3d.push(peak);
                }
            }
        }

        // 13C-HSQC-TOCSY 3D: (C, H_anchor, H_tocsy)
        if filter.should_include("hsqc-tocsy-13c-3d") {
            for (carbon_name, c_shift, c_ref, direct_protons) in &residue_carbons {
                // Find the first attached proton to use as anchor
                let h_anchor_opt = direct_protons.iter()
                    .find_map(|&p| residue_protons.iter().find(|(name, _, _)| name == p));

                if let Some((_, h_anchor_shift, h_anchor_ref)) = h_anchor_opt {
                    for (h_name, h_shift, h_ref) in &residue_protons {
                        let is_anchor = direct_protons.iter().any(|&p| p == h_name);
                        let intensity = if is_anchor {
                            1.0
                        } else if h_name == "H" {
                            0.7
                        } else {
                            0.6
                        };

                        let peak = UnlabeledPeak::hsqc_tocsy_13c_3d(*c_shift, *h_anchor_shift, *h_shift, intensity);
                        let atom_desc = format!("{}/{}/{}", carbon_name, direct_protons.first().unwrap_or(&"H"), h_name);
                        let reference = vec![
                            ("C".to_string(), *c_ref),         // KDE mode for C
                            ("H_anchor".to_string(), *h_anchor_ref), // KDE mode for anchor H
                            ("H".to_string(), *h_ref),         // KDE mode for TOCSY H
                        ];
                        ground_truth.register_with_reference(&peak, seq_code, &atom_desc, reference);
                        hsqc_tocsy_13c_3d.push(peak);
                    }
                }
            }
        }
    }

    // Sequential NOESY: H(i) to HA(i-1)
    if filter.should_include("noesy") {
        for i in 2..=sequence.len() as i32 {
            let curr_seq = i;
            let prev_seq = i - 1;

            if let (Some(&h_curr), Some(&ha_prev), Some(&h_ref), Some(&ha_ref)) =
                (h_shifts.get(&curr_seq), ha_shifts.get(&prev_seq),
                 h_refs.get(&curr_seq), ha_refs.get(&prev_seq))
            {
                let peak = UnlabeledPeak::noesy(h_curr, ha_prev, 1.0);
                // Register with current residue (where backbone H is)
                let atom_desc = format!("H(i)/HA(i-1)");
                let reference = vec![
                    ("H1".to_string(), h_ref),   // KDE mode for H(i)
                    ("H2".to_string(), ha_ref),  // KDE mode for HA(i-1)
                ];
                ground_truth.register_with_reference(&peak, curr_seq, &atom_desc, reference);
                noesy.push(peak);
            }
        }
    }

    // 3D Triple-Resonance Experiments
    // These provide unambiguous sequential connectivity via CA/CB chemical shift matching
    let seq_chars: Vec<char> = sequence.chars().collect();

    for i in 2..=sequence.len() as i32 {
        let curr_seq = i;
        let prev_seq = i - 1;

        // Skip proline at current position (no backbone NH)
        let curr_aa = seq_chars.get((curr_seq - 1) as usize).copied().unwrap_or('X');
        if curr_aa == 'P' {
            continue;
        }

        // Get backbone H and N for current residue (observed values)
        let (h_val, n_val) = match (h_shifts.get(&curr_seq), n_shifts.get(&curr_seq)) {
            (Some(&h), Some(&n)) => (h, n),
            _ => continue,
        };
        // Get reference values (KDE modes)
        let h_ref_val = h_refs.get(&curr_seq).copied();
        let n_ref_val = n_refs.get(&curr_seq).copied();

        // Get carbon/proton shifts for current and previous residues (observed)
        let ca_curr = ca_shifts.get(&curr_seq).copied();
        let cb_curr = cb_shifts.get(&curr_seq).copied();
        let ca_prev = ca_shifts.get(&prev_seq).copied();
        let cb_prev = cb_shifts.get(&prev_seq).copied();
        let co_prev = co_shifts.get(&prev_seq).copied();
        let ha_prev_val = ha_shifts.get(&prev_seq).copied();
        let hb_prev_val = hb_shifts.get(&prev_seq).copied();
        // Reference values (KDE modes) for carbons/protons
        let ca_curr_ref = ca_refs.get(&curr_seq).copied();
        let cb_curr_ref = cb_refs.get(&curr_seq).copied();
        let ca_prev_ref = ca_refs.get(&prev_seq).copied();
        let cb_prev_ref = cb_refs.get(&prev_seq).copied();
        let co_prev_ref = co_refs.get(&prev_seq).copied();
        let ha_prev_ref = ha_refs.get(&prev_seq).copied();
        let hb_prev_ref = hb_refs.get(&prev_seq).copied();

        // HNCO: (H, N, CO) - correlates NH(i) with CO(i-1)
        // Ground truth: the CO belongs to residue i-1
        if filter.should_include("hnco") {
            if let (Some(co), Some(h_r), Some(n_r), Some(co_r)) =
                (co_prev, h_ref_val, n_ref_val, co_prev_ref)
            {
                let peak = UnlabeledPeak::hnco(h_val, n_val, co, 1.0);
                let reference = vec![
                    ("H".to_string(), h_r),
                    ("N".to_string(), n_r),
                    ("C".to_string(), co_r),
                ];
                ground_truth.register_with_reference(&peak, curr_seq - 1, "H/N/C(i-1)", reference);
                hnco.push(peak);
            }
        }

        // HNCA: (H, N, CA) - correlates NH(i) with CA(i) strong, CA(i-1) weak
        if filter.should_include("hnca") {
            // Intra-residue CA(i) - strong
            if let (Some(ca), Some(h_r), Some(n_r), Some(ca_r)) =
                (ca_curr, h_ref_val, n_ref_val, ca_curr_ref)
            {
                let peak = UnlabeledPeak::hnca(h_val, n_val, ca, 1.0);
                let reference = vec![
                    ("H".to_string(), h_r),
                    ("N".to_string(), n_r),
                    ("CA".to_string(), ca_r),
                ];
                ground_truth.register_with_reference(&peak, curr_seq, "H/N/CA(i)", reference);
                hnca.push(peak);
            }
            // Inter-residue CA(i-1) - weak (~30%)
            if let (Some(ca), Some(h_r), Some(n_r), Some(ca_r)) =
                (ca_prev, h_ref_val, n_ref_val, ca_prev_ref)
            {
                let peak = UnlabeledPeak::hnca(h_val, n_val, ca, 0.3);
                // Ground truth: the carbon BELONGS to residue i-1
                let reference = vec![
                    ("H".to_string(), h_r),
                    ("N".to_string(), n_r),
                    ("CA".to_string(), ca_r),
                ];
                ground_truth.register_with_reference(&peak, curr_seq - 1, "H/N/CA(i-1)", reference);
                hnca.push(peak);
            }
        }

        // HNCACB: (H, N, C) - positive for intra (i), negative for inter (i-1)
        if filter.should_include("hncacb") {
            // Intra-residue CA(i) - positive
            if let (Some(ca), Some(h_r), Some(n_r), Some(ca_r)) =
                (ca_curr, h_ref_val, n_ref_val, ca_curr_ref)
            {
                let peak = UnlabeledPeak::hncacb(h_val, n_val, ca, 1.0);
                let reference = vec![
                    ("H".to_string(), h_r),
                    ("N".to_string(), n_r),
                    ("CA".to_string(), ca_r),
                ];
                ground_truth.register_with_reference(&peak, curr_seq, "H/N/CA(i)+", reference);
                hncacb.push(peak);
            }
            // Intra-residue CB(i) - positive
            if let (Some(cb), Some(h_r), Some(n_r), Some(cb_r)) =
                (cb_curr, h_ref_val, n_ref_val, cb_curr_ref)
            {
                let peak = UnlabeledPeak::hncacb(h_val, n_val, cb, 0.8);
                let reference = vec![
                    ("H".to_string(), h_r),
                    ("N".to_string(), n_r),
                    ("CB".to_string(), cb_r),
                ];
                ground_truth.register_with_reference(&peak, curr_seq, "H/N/CB(i)+", reference);
                hncacb.push(peak);
            }
            // Inter-residue CA(i-1) - negative
            // Ground truth: the carbon BELONGS to residue i-1, not current backbone
            if let (Some(ca), Some(h_r), Some(n_r), Some(ca_r)) =
                (ca_prev, h_ref_val, n_ref_val, ca_prev_ref)
            {
                let peak = UnlabeledPeak::hncacb(h_val, n_val, ca, -0.3);
                let reference = vec![
                    ("H".to_string(), h_r),
                    ("N".to_string(), n_r),
                    ("CA".to_string(), ca_r),
                ];
                ground_truth.register_with_reference(&peak, curr_seq - 1, "H/N/CA(i-1)-", reference);
                hncacb.push(peak);
            }
            // Inter-residue CB(i-1) - negative
            if let (Some(cb), Some(h_r), Some(n_r), Some(cb_r)) =
                (cb_prev, h_ref_val, n_ref_val, cb_prev_ref)
            {
                let peak = UnlabeledPeak::hncacb(h_val, n_val, cb, -0.25);
                let reference = vec![
                    ("H".to_string(), h_r),
                    ("N".to_string(), n_r),
                    ("CB".to_string(), cb_r),
                ];
                ground_truth.register_with_reference(&peak, curr_seq - 1, "H/N/CB(i-1)-", reference);
                hncacb.push(peak);
            }
        }

        // CBCACONH: (H, N, CA/CB) - only i-1 carbons
        // Ground truth: ALL peaks show previous residue's carbons
        if filter.should_include("cbcaconh") {
            if let (Some(ca), Some(h_r), Some(n_r), Some(ca_r)) =
                (ca_prev, h_ref_val, n_ref_val, ca_prev_ref)
            {
                let peak = UnlabeledPeak::cbcaconh(h_val, n_val, ca, 1.0);
                let reference = vec![
                    ("H".to_string(), h_r),
                    ("N".to_string(), n_r),
                    ("CA".to_string(), ca_r),
                ];
                ground_truth.register_with_reference(&peak, curr_seq - 1, "H/N/CA(i-1)", reference);
                cbcaconh.push(peak);
            }
            if let (Some(cb), Some(h_r), Some(n_r), Some(cb_r)) =
                (cb_prev, h_ref_val, n_ref_val, cb_prev_ref)
            {
                let peak = UnlabeledPeak::cbcaconh(h_val, n_val, cb, 0.8);
                let reference = vec![
                    ("H".to_string(), h_r),
                    ("N".to_string(), n_r),
                    ("CB".to_string(), cb_r),
                ];
                ground_truth.register_with_reference(&peak, curr_seq - 1, "H/N/CB(i-1)", reference);
                cbcaconh.push(peak);
            }
        }

        // HBHACONH: (H, N, HA/HB) - only i-1 protons
        if filter.should_include("hbhaconh") {
            if let (Some(ha), Some(h_r), Some(n_r), Some(ha_r)) =
                (ha_prev_val, h_ref_val, n_ref_val, ha_prev_ref)
            {
                let peak = UnlabeledPeak::hbhaconh(h_val, n_val, ha, 1.0);
                let reference = vec![
                    ("H".to_string(), h_r),
                    ("N".to_string(), n_r),
                    ("HA".to_string(), ha_r),
                ];
                ground_truth.register_with_reference(&peak, curr_seq, "H/N/HA(i-1)", reference);
                hbhaconh.push(peak);
            }
            if let (Some(hb), Some(h_r), Some(n_r), Some(hb_r)) =
                (hb_prev_val, h_ref_val, n_ref_val, hb_prev_ref)
            {
                let peak = UnlabeledPeak::hbhaconh(h_val, n_val, hb, 0.8);
                let reference = vec![
                    ("H".to_string(), h_r),
                    ("N".to_string(), n_r),
                    ("HB".to_string(), hb_r),
                ];
                ground_truth.register_with_reference(&peak, curr_seq, "H/N/HB(i-1)", reference);
                hbhaconh.push(peak);
            }
        }
    }

    (hsqc_15n, hsqc_13c, tocsy, noesy, hsqc_tocsy_15n, hsqc_tocsy_13c, hsqc_tocsy_15n_3d, hsqc_tocsy_13c_3d, hnco, hnca, hncacb, cbcaconh, hbhaconh, ground_truth)
}

/// Evaluate assignment results against ground truth
fn evaluate_results(results: &[UnifiedAssignmentResult], ground_truth: &GroundTruth, sequence: &str) -> TestSummary {
    let mut total_peaks = 0;
    let mut correct = 0;
    let mut backbone_peaks = 0;
    let mut backbone_correct = 0;
    let mut peak_results = Vec::new();
    let mut per_aa: HashMap<char, AAStats> = HashMap::new();

    let seq_chars: Vec<char> = sequence.chars().collect();

    for result in results {
        if let Some(&actual_residue) = ground_truth.peak_to_residue.get(&result.peak_id) {
            total_peaks += 1;
            let is_correct = result.assigned_residue == actual_residue;

            // Get amino acid for this residue (1-indexed)
            let aa = if actual_residue > 0 && (actual_residue as usize) <= seq_chars.len() {
                seq_chars[actual_residue as usize - 1]
            } else {
                '?'
            };

            // Update per-AA stats
            let aa_stats = per_aa.entry(aa).or_default();
            aa_stats.total += 1;
            if is_correct {
                aa_stats.correct += 1;
            }

            if is_correct {
                correct += 1;
            }

            if matches!(result.peak_type, PeakType::Backbone) {
                backbone_peaks += 1;
                aa_stats.backbone_total += 1;
                if is_correct {
                    backbone_correct += 1;
                    aa_stats.backbone_correct += 1;
                }
            }

            let atom_name = ground_truth
                .peak_to_atom
                .get(&result.peak_id)
                .cloned()
                .unwrap_or_else(|| "?".to_string());

            // Get shifts from ground truth
            let shifts = ground_truth
                .peak_to_shifts
                .get(&result.peak_id)
                .cloned()
                .unwrap_or_default();

            // Get reference shifts (for synthetic data)
            let reference_shifts = ground_truth
                .peak_to_reference
                .get(&result.peak_id)
                .cloned()
                .unwrap_or_default();

            peak_results.push(PeakResult {
                peak_id: result.peak_id,
                predicted_residue: result.assigned_residue,
                actual_residue,
                confidence: result.confidence,
                is_correct,
                peak_type: result.peak_type.clone(),
                atom_name,
                shifts,
                reference_shifts,
            });
        }
    }

    // Sort by residue for readability
    peak_results.sort_by_key(|r| (r.actual_residue, r.atom_name.clone()));

    TestSummary {
        total_peaks,
        correct,
        backbone_peaks,
        backbone_correct,
        results: peak_results,
        per_aa,
    }
}

/// Evaluate observation-based assignment results against ground truth
fn evaluate_observation_results(
    results: &[ObservationAssignmentResult],
    ground_truth: &GroundTruth,
    sequence: &str,
) -> TestSummary {
    let mut total_peaks = 0;
    let mut correct = 0;
    let mut backbone_peaks = 0;
    let mut backbone_correct = 0;
    let mut peak_results = Vec::new();
    let mut per_aa: HashMap<char, AAStats> = HashMap::new();

    let seq_chars: Vec<char> = sequence.chars().collect();

    for result in results {
        if let Some(&actual_residue) = ground_truth.peak_to_residue.get(&result.observation_id) {
            total_peaks += 1;
            let is_correct = result.assigned_residue == actual_residue;

            // Get amino acid for this residue (1-indexed)
            let aa = if actual_residue > 0 && (actual_residue as usize) <= seq_chars.len() {
                seq_chars[actual_residue as usize - 1]
            } else {
                '?'
            };

            // Update per-AA stats
            let aa_stats = per_aa.entry(aa).or_default();
            aa_stats.total += 1;
            if is_correct {
                aa_stats.correct += 1;
            }

            if is_correct {
                correct += 1;
            }

            let atom_name = ground_truth
                .peak_to_atom
                .get(&result.observation_id)
                .cloned()
                .unwrap_or_else(|| "?".to_string());

            // Map experiment type to PeakType for display
            let peak_type = match result.experiment_type {
                nmraster_lib::data::spin_system::PeakExperimentType::Hsqc15N => PeakType::Backbone,
                nmraster_lib::data::spin_system::PeakExperimentType::Hsqc13C => PeakType::Carbon,
                _ => PeakType::Carbon, // TOCSY, NOESY etc. treated as Carbon for display
            };

            // Count backbone peaks (15N-HSQC)
            if matches!(peak_type, PeakType::Backbone) {
                backbone_peaks += 1;
                aa_stats.backbone_total += 1;
                if is_correct {
                    backbone_correct += 1;
                    aa_stats.backbone_correct += 1;
                }
            }

            // Get shifts from ground truth
            let shifts = ground_truth
                .peak_to_shifts
                .get(&result.observation_id)
                .cloned()
                .unwrap_or_default();

            // Get reference shifts (for synthetic data)
            let reference_shifts = ground_truth
                .peak_to_reference
                .get(&result.observation_id)
                .cloned()
                .unwrap_or_default();

            peak_results.push(PeakResult {
                peak_id: result.observation_id,
                predicted_residue: result.assigned_residue,
                actual_residue,
                confidence: result.confidence,
                is_correct,
                peak_type,
                atom_name,
                shifts,
                reference_shifts,
            });
        }
    }

    // Sort by residue for readability
    peak_results.sort_by_key(|r| (r.actual_residue, r.atom_name.clone()));

    TestSummary {
        total_peaks,
        correct,
        backbone_peaks,
        backbone_correct,
        results: peak_results,
        per_aa,
    }
}

/// Print per-residue data density from BMRB shifts
fn print_data_density(shifts: &[DepositedShift], sequence: &str) {
    let mut by_residue: HashMap<i32, Vec<&DepositedShift>> = HashMap::new();
    for shift in shifts {
        by_residue
            .entry(shift.residue_seq_code)
            .or_default()
            .push(shift);
    }

    println!("\nData density per residue:");
    println!("  Pos | AA | Atoms | H  | C  | N  | Protons");
    println!("  {}", "-".repeat(45));

    let seq_chars: Vec<char> = sequence.chars().collect();
    let mut sparse_residues = Vec::new();

    for (i, aa) in seq_chars.iter().enumerate() {
        let pos = (i + 1) as i32;
        if let Some(res_shifts) = by_residue.get(&pos) {
            let total = res_shifts.len();
            let h_count = res_shifts.iter().filter(|s| s.atom_name.starts_with('H')).count();
            let c_count = res_shifts.iter().filter(|s| s.atom_name.starts_with('C')).count();
            let n_count = res_shifts.iter().filter(|s| s.atom_name.starts_with('N')).count();

            let marker = if total < 8 { " !" } else { "" };
            if total < 8 {
                sparse_residues.push((*aa, pos));
            }

            println!("  {:3} | {} | {:5} | {:2} | {:2} | {:2} | {:7}{}",
                pos, aa, total, h_count, c_count, n_count, h_count, marker);
        } else {
            println!("  {:3} | {} | {:5} | -- | -- | -- | -- (NO DATA!)", pos, aa, 0);
            sparse_residues.push((*aa, pos));
        }
    }

    if !sparse_residues.is_empty() {
        println!("\n  Sparse residues (<8 atoms): {:?}", sparse_residues);
    }
    println!();
}

/// Print results table and summary
fn print_results(summary: &TestSummary, sequence: &str, mode_desc: &str) {
    let width = 79;

    println!();
    println!("{}", "=".repeat(width));
    println!("{:^width$}", "UNIFIED ASSIGNMENT TEST RESULTS");
    println!("{}", "=".repeat(width));
    println!();
    println!("Sequence: {} ({} residues)", sequence, sequence.len());
    println!("Mode: {}", mode_desc);
    println!();
    println!("{}", "-".repeat(width));
    println!(
        " {:8} | {:4} | {:9} | {:6} | {:5} | {:10} | {:8}",
        "Peak ID", "Type", "Predicted", "Actual", "Match", "Confidence", "Atom"
    );
    println!("{}", "-".repeat(width));

    for result in &summary.results {
        let type_str = match result.peak_type {
            PeakType::Backbone => "BB",
            PeakType::Carbon => "C",
        };

        let match_str = if result.is_correct { "OK" } else { "MISS" };

        println!(
            " {:8} | {:4} | {:9} | {:6} | {:5} | {:9.1}% | {:8}",
            &result.peak_id.to_string()[..8],
            type_str,
            result.predicted_residue,
            result.actual_residue,
            match_str,
            result.confidence * 100.0,
            result.atom_name
        );
    }

    println!("{}", "-".repeat(width));
    println!();
    println!("SUMMARY:");
    println!("  Total peaks:       {}", summary.total_peaks);
    println!("  Correct:           {}", summary.correct);

    let overall_acc = if summary.total_peaks > 0 {
        (summary.correct as f64 / summary.total_peaks as f64) * 100.0
    } else {
        0.0
    };
    println!("  Overall accuracy:  {:.1}%", overall_acc);

    println!();
    println!("  Backbone peaks:    {}", summary.backbone_peaks);
    println!("  Backbone correct:  {}", summary.backbone_correct);

    let bb_acc = if summary.backbone_peaks > 0 {
        (summary.backbone_correct as f64 / summary.backbone_peaks as f64) * 100.0
    } else {
        0.0
    };
    println!("  Backbone accuracy: {:.1}%", bb_acc);

    // Per amino acid breakdown
    if !summary.per_aa.is_empty() {
        println!();
        println!("PER AMINO ACID:");
        println!("  {:3} | {:5} | {:7} | {:8}", "AA", "Total", "Correct", "Accuracy");
        println!("  {}", "-".repeat(35));

        // Sort by accuracy (worst first) to highlight problematic AAs
        let mut aa_list: Vec<_> = summary.per_aa.iter().collect();
        aa_list.sort_by(|a, b| {
            let acc_a = if a.1.total > 0 { a.1.correct as f64 / a.1.total as f64 } else { 0.0 };
            let acc_b = if b.1.total > 0 { b.1.correct as f64 / b.1.total as f64 } else { 0.0 };
            acc_a.partial_cmp(&acc_b).unwrap()
        });

        for (aa, stats) in aa_list {
            let acc = if stats.total > 0 {
                (stats.correct as f64 / stats.total as f64) * 100.0
            } else {
                0.0
            };
            let marker = if acc < 50.0 { " !" } else { "" };
            println!("  {:3} | {:5} | {:7} | {:6.1}%{}", aa, stats.total, stats.correct, acc, marker);
        }
    }
    println!();

    // Print atom-centric table at the end
    print_atom_table(summary, sequence);
}

/// Print atom-centric results table organized by residue
fn print_atom_table(summary: &TestSummary, sequence: &str) {
    let seq_chars: Vec<char> = sequence.chars().collect();
    let width = 105;

    println!("{}", "=".repeat(width));
    println!("{:^width$}", "ATOM-BY-ATOM ASSIGNMENT TABLE");
    println!("{}", "=".repeat(width));
    println!();
    println!(
        " {:3} | {:3} | {:14} | {:>18} | {:>18} | {:>8} | {:6}",
        "Res", "AA", "Atom", "Observed", "Reference", "Assigned", "Match"
    );
    println!("{}", "-".repeat(width));

    // Group results by actual residue
    let mut by_residue: HashMap<i32, Vec<&PeakResult>> = HashMap::new();
    for result in &summary.results {
        by_residue
            .entry(result.actual_residue)
            .or_default()
            .push(result);
    }

    // Print in residue order
    let mut residues: Vec<_> = by_residue.keys().copied().collect();
    residues.sort();

    for res in residues {
        let results = &by_residue[&res];
        let aa = if res > 0 && (res as usize) <= seq_chars.len() {
            seq_chars[res as usize - 1]
        } else {
            '?'
        };

        // Sort by atom name within each residue
        let mut sorted_results: Vec<_> = results.iter().copied().collect();
        sorted_results.sort_by(|a, b| a.atom_name.cmp(&b.atom_name));

        for result in sorted_results {
            let match_str = if result.is_correct { "OK" } else { "MISS" };

            // Format observed shifts - show key shift (last one is usually most distinctive)
            let observed_str = format_key_shift(&result.shifts);

            // Format reference shifts
            let reference_str = if result.reference_shifts.is_empty() {
                "-".to_string()
            } else {
                format_key_shift(&result.reference_shifts)
            };

            println!(
                " {:3} | {:3} | {:14} | {:>18} | {:>18} | {:>8} | {:6}",
                res,
                aa,
                result.atom_name,
                observed_str,
                reference_str,
                result.predicted_residue,
                match_str
            );
        }
    }

    println!("{}", "-".repeat(width));
    println!();
}

/// Print chemical shift table: for each atom position, show expected vs assigned shift
/// This is the user's desired format - atom-centric, not peak-centric
fn print_chemical_shift_table(
    results: &[ObservationAssignmentResult],
    ground_truth: &GroundTruth,
    sequence: &str,
) {
    let seq_chars: Vec<char> = sequence.chars().collect();
    let width = 80;

    println!();
    println!("{}", "=".repeat(width));
    println!("{:^width$}", "CHEMICAL SHIFT TABLE (Expected vs Assigned)");
    println!("{}", "=".repeat(width));
    println!();
    println!(
        " {:3} | {:3} | {:6} | {:>10} | {:>10} | {:>8} | {}",
        "Res", "AA", "Atom", "Expected", "Assigned", "Delta", "Match"
    );
    println!("{}", "-".repeat(width));

    // Build a map: for each (residue, atom_name) -> assigned shift value
    // We find this by looking at peaks that were assigned to each residue
    // and matching their atom types
    let mut assigned_shifts: HashMap<(i32, String), f64> = HashMap::new();

    for result in results {
        let assigned_res = result.assigned_residue;
        if assigned_res <= 0 {
            continue; // Unassigned
        }

        // Get the atom name for this peak
        if let Some(atom_desc) = ground_truth.peak_to_atom.get(&result.observation_id) {
            // Get the shifts from this peak
            if let Some(shifts) = ground_truth.peak_to_shifts.get(&result.observation_id) {
                // Parse atom description like "H/N", "CA/HA", or just "H1/H2" for TOCSY
                for (nucleus_name, shift_val) in shifts {
                    // Map nucleus name to atom name based on context
                    let atom_name = map_nucleus_to_atom(nucleus_name, atom_desc);
                    let key = (assigned_res, atom_name);
                    // Store the shift (if multiple peaks assign same atom, last wins)
                    assigned_shifts.insert(key, *shift_val);
                }
            }
        }
    }

    // Collect all expected atoms and sort by (residue, atom_name)
    let mut atoms: Vec<_> = ground_truth.expected_atom_shifts.keys().collect();
    atoms.sort_by(|a, b| {
        a.0.cmp(&b.0).then_with(|| atom_sort_key(&a.1).cmp(&atom_sort_key(&b.1)))
    });

    let mut total_atoms = 0;
    let mut correct_atoms = 0;
    let mut current_res = -1;

    for (residue, atom_name) in atoms {
        let expected = ground_truth.expected_atom_shifts.get(&(*residue, atom_name.clone())).copied().unwrap_or(0.0);

        // Get amino acid
        let aa = if *residue > 0 && (*residue as usize) <= seq_chars.len() {
            seq_chars[*residue as usize - 1]
        } else {
            '?'
        };

        // Look up assigned shift
        let assigned = assigned_shifts.get(&(*residue, atom_name.clone())).copied();

        // Add separator line between residues
        if *residue != current_res && current_res > 0 {
            println!("{}", "-".repeat(width));
        }
        current_res = *residue;

        total_atoms += 1;

        if let Some(assigned_val) = assigned {
            let delta = (assigned_val - expected).abs();
            // Tolerance depends on nucleus type
            let tolerance = if atom_name.starts_with('H') { 0.1 } else if atom_name.starts_with('N') { 1.0 } else { 1.0 };
            let is_match = delta < tolerance;
            if is_match {
                correct_atoms += 1;
            }
            let match_str = if is_match { "OK" } else { "MISS" };

            println!(
                " {:3} | {:3} | {:6} | {:>10.3} | {:>10.3} | {:>8.3} | {}",
                residue, aa, atom_name, expected, assigned_val, delta, match_str
            );
        } else {
            // No assignment found for this atom
            println!(
                " {:3} | {:3} | {:6} | {:>10.3} | {:>10} | {:>8} | {}",
                residue, aa, atom_name, expected, "-", "-", "MISS"
            );
        }
    }

    println!("{}", "-".repeat(width));
    println!();

    // Summary
    let accuracy = if total_atoms > 0 {
        100.0 * correct_atoms as f64 / total_atoms as f64
    } else {
        0.0
    };
    println!("Atom-level accuracy: {}/{} ({:.1}%)", correct_atoms, total_atoms, accuracy);
    println!();
}

/// Map nucleus name (from peak) to atom name based on atom description
fn map_nucleus_to_atom(nucleus: &str, atom_desc: &str) -> String {
    // atom_desc examples: "H/N", "CA/HA", "CB/HB2", "H1/H2" (TOCSY)
    let parts: Vec<&str> = atom_desc.split('/').collect();

    match nucleus {
        "H" | "HN" => {
            // Backbone amide proton
            "H".to_string()
        }
        "N" | "N15" => "N".to_string(),
        "CA" | "C13_CA" => "CA".to_string(),
        "CB" | "C13_CB" => "CB".to_string(),
        "CO" | "C13_CO" => "C".to_string(), // Carbonyl
        "C" => {
            // Could be CA, CB, etc - try to get from atom_desc
            if parts.len() >= 1 && parts[0].starts_with('C') {
                parts[0].to_string()
            } else {
                "C".to_string()
            }
        }
        "HA" | "HA2" | "HA3" => nucleus.to_string(),
        "HB" | "HB2" | "HB3" => nucleus.to_string(),
        "H1" => {
            // TOCSY - first proton, try to get from atom_desc
            if parts.len() >= 1 {
                parts[0].to_string()
            } else {
                nucleus.to_string()
            }
        }
        "H2" => {
            // TOCSY - second proton
            if parts.len() >= 2 {
                parts[1].to_string()
            } else {
                nucleus.to_string()
            }
        }
        _ => nucleus.to_string()
    }
}

/// Sort key for atoms: backbone first (H, N, CA, C), then sidechains
fn atom_sort_key(atom: &str) -> (u8, String) {
    let order = match atom {
        "H" => 0,
        "N" => 1,
        "CA" => 2,
        "C" => 3,  // Carbonyl
        "HA" | "HA2" | "HA3" => 4,
        "CB" => 5,
        "HB" | "HB2" | "HB3" => 6,
        _ if atom.starts_with("CG") => 7,
        _ if atom.starts_with("HG") => 8,
        _ if atom.starts_with("CD") => 9,
        _ if atom.starts_with("HD") => 10,
        _ if atom.starts_with("CE") => 11,
        _ if atom.starts_with("HE") => 12,
        _ if atom.starts_with("CZ") => 13,
        _ if atom.starts_with("HZ") => 14,
        _ => 20,
    };
    (order, atom.to_string())
}

/// Format key shifts for display (show most distinctive ones)
fn format_key_shift(shifts: &[(String, f64)]) -> String {
    if shifts.is_empty() {
        return "-".to_string();
    }

    // For 2D experiments, show both
    // For 3D experiments, focus on key nuclei (C, N rather than H)
    let key_shifts: Vec<_> = shifts.iter()
        .filter(|(name, _)| {
            // Prioritize carbon and nitrogen shifts (more distinctive)
            name.starts_with('C') || name.starts_with('N') ||
            name == "H" || name == "H1" || name == "H2"
        })
        .take(3)  // Max 3 shifts to fit
        .collect();

    if key_shifts.is_empty() {
        // Fallback to first shifts
        shifts.iter()
            .take(2)
            .map(|(name, val)| format!("{}={:.1}", name, val))
            .collect::<Vec<_>>()
            .join(", ")
    } else {
        key_shifts.iter()
            .map(|(name, val)| format!("{}={:.1}", name, val))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Build JSON output from test results
fn build_json_output(
    entry_id: u32,
    residue_range: Option<String>,
    sequence: &str,
    timing_ms: f64,
    summary: &TestSummary,
    hsqc_15n_count: usize,
    hsqc_13c_count: usize,
    tocsy_count: usize,
    noesy_count: usize,
) -> JsonOutput {
    let overall_accuracy = if summary.total_peaks > 0 {
        summary.correct as f64 / summary.total_peaks as f64
    } else {
        0.0
    };

    let backbone_accuracy = if summary.backbone_peaks > 0 {
        summary.backbone_correct as f64 / summary.backbone_peaks as f64
    } else {
        0.0
    };

    // Convert per-AA stats
    let mut per_amino_acid = HashMap::new();
    let seq_chars: Vec<char> = sequence.chars().collect();

    for (&aa, stats) in &summary.per_aa {
        let accuracy = if stats.total > 0 {
            stats.correct as f64 / stats.total as f64
        } else {
            0.0
        };
        per_amino_acid.insert(
            aa.to_string(),
            JsonAAStats {
                total: stats.total,
                correct: stats.correct,
                accuracy,
            },
        );
    }

    // Collect failures
    let failures: Vec<JsonFailure> = summary
        .results
        .iter()
        .filter(|r| !r.is_correct)
        .map(|r| {
            let aa = if r.actual_residue > 0 && (r.actual_residue as usize) <= seq_chars.len() {
                seq_chars[r.actual_residue as usize - 1]
            } else {
                '?'
            };
            JsonFailure {
                residue: r.actual_residue,
                amino_acid: aa,
                predicted: r.predicted_residue,
                actual: r.actual_residue,
                atom: r.atom_name.clone(),
                confidence: r.confidence,
            }
        })
        .collect();

    JsonOutput {
        entry_id,
        residue_range,
        sequence: sequence.to_string(),
        num_residues: sequence.len(),
        timing_ms,
        overall_accuracy,
        backbone_accuracy,
        peak_counts: PeakCounts {
            hsqc_15n: hsqc_15n_count,
            hsqc_13c: hsqc_13c_count,
            tocsy: tocsy_count,
            noesy: noesy_count,
            total: hsqc_15n_count + hsqc_13c_count + tocsy_count + noesy_count,
        },
        per_amino_acid,
        failures,
    }
}

/// Run batch tests across multiple residue windows
fn run_batch_mode(entry_id: u32, output_dir: PathBuf, windows: &str, verbose: bool, filter: &ExperimentFilter) {
    println!("Running batch tests for BMRB entry {}...", entry_id);

    // Parse window sizes
    let window_sizes: Vec<usize> = windows
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if window_sizes.is_empty() {
        eprintln!("No valid window sizes specified");
        std::process::exit(1);
    }

    // Create output directory
    if let Err(e) = std::fs::create_dir_all(&output_dir) {
        eprintln!("Error creating output directory: {}", e);
        std::process::exit(1);
    }

    // Fetch all shifts
    let shifts = match fetch_entry_shifts(entry_id) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error fetching BMRB entry: {}", e);
            std::process::exit(1);
        }
    };

    let full_sequence = extract_sequence(&shifts);
    let total_residues = full_sequence.len();
    println!("Full sequence: {} ({} residues)", full_sequence, total_residues);

    // For each window size, run multiple tests at different positions
    for window_size in &window_sizes {
        let window_size = *window_size;
        if window_size > total_residues {
            println!("Skipping window size {} (larger than sequence)", window_size);
            continue;
        }

        println!("\n--- Window size: {} ---", window_size);

        // Test at multiple positions: start, middle regions, end
        let positions: Vec<usize> = if window_size >= total_residues {
            vec![1] // Just run full sequence
        } else {
            let mut pos = vec![1]; // Start

            // Add middle positions (every 1/3 of the way through)
            let step = (total_residues - window_size) / 3;
            if step > 0 {
                pos.push(1 + step);
                pos.push(1 + 2 * step);
            }

            // End
            pos.push(total_residues - window_size + 1);
            pos.sort();
            pos.dedup();
            pos
        };

        for start in positions {
            let end = start + window_size - 1;
            let range = format!("{}-{}", start, end);

            println!("  Testing residues {}...", range);

            // Filter shifts for this range
            let mut window_shifts = match filter_residue_range(&shifts, &range) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("    Error filtering residues: {}", e);
                    continue;
                }
            };
            renumber_shifts(&mut window_shifts);

            if window_shifts.is_empty() {
                println!("    No shifts found");
                continue;
            }

            let sequence = extract_sequence(&window_shifts);

            // Generate peaks
            let (hsqc_15n, hsqc_13c, tocsy, noesy, hsqc_tocsy_15n, hsqc_tocsy_13c, hsqc_tocsy_15n_3d, hsqc_tocsy_13c_3d, hnco, hnca, hncacb, cbcaconh, hbhaconh, ground_truth) =
                generate_peaks_from_bmrb(&window_shifts, filter);

            // Run assignment with timing
            let params = UnifiedAssignmentParams {
                verbose,
                ..Default::default()
            };

            let start_time = Instant::now();
            let results = run_unified_assignment(
                &hsqc_15n, &hsqc_13c, &tocsy, &noesy, &hsqc_tocsy_15n, &hsqc_tocsy_13c,
                &hsqc_tocsy_15n_3d, &hsqc_tocsy_13c_3d,
                &hnco, &hnca, &hncacb, &cbcaconh, &hbhaconh,
                &sequence, &params
            );
            let elapsed = start_time.elapsed();
            let timing_ms = elapsed.as_secs_f64() * 1000.0;

            // Evaluate
            let summary = evaluate_results(&results, &ground_truth, &sequence);

            let overall_acc = if summary.total_peaks > 0 {
                (summary.correct as f64 / summary.total_peaks as f64) * 100.0
            } else {
                0.0
            };

            let bb_acc = if summary.backbone_peaks > 0 {
                (summary.backbone_correct as f64 / summary.backbone_peaks as f64) * 100.0
            } else {
                0.0
            };

            println!(
                "    Overall: {:.1}%, Backbone: {:.1}%, Time: {:.0}ms",
                overall_acc, bb_acc, timing_ms
            );

            // Write JSON output
            let json_output = build_json_output(
                entry_id,
                Some(range.clone()),
                &sequence,
                timing_ms,
                &summary,
                hsqc_15n.len(),
                hsqc_13c.len(),
                tocsy.len(),
                noesy.len(),
            );

            let json_filename = format!("entry{}_w{}_r{}.json", entry_id, window_size, range.replace("-", "_"));
            let json_path = output_dir.join(&json_filename);

            match serde_json::to_string_pretty(&json_output) {
                Ok(json_str) => {
                    if let Err(e) = std::fs::write(&json_path, json_str) {
                        eprintln!("    Error writing JSON: {}", e);
                    }
                }
                Err(e) => eprintln!("    Error serializing JSON: {}", e),
            }
        }
    }

    println!("\nBatch tests complete. Results written to: {}", output_dir.display());
}

// ============================================================================
// GRID SEARCH IMPLEMENTATION
// ============================================================================

/// Grid A: Weight ratio configurations (9 total)
/// Format: (tocsy_init, tocsy_final, typing_init, typing_final, sequential, seq_type)
const GRID_A: &[(f64, f64, f64, f64, f64, f64)] = &[
    // A1: Keep TOCSY constant
    (2.0, 2.0, 2.0, 4.0, 1.0, 10.0),
    // A2: TOCSY dominant
    (3.0, 2.0, 2.0, 3.0, 1.0, 10.0),
    // A3: Typing dominant
    (1.5, 1.5, 4.0, 6.0, 1.0, 10.0),
    // A4: Strong sequential
    (2.0, 2.0, 2.0, 4.0, 2.0, 10.0),
    // A5: Weak seq-type
    (2.0, 2.0, 2.0, 4.0, 1.0, 5.0),
    // A6: Strong seq-type
    (2.0, 2.0, 2.0, 4.0, 1.0, 20.0),
    // A7: TOCSY very strong
    (4.0, 3.0, 1.0, 2.0, 1.5, 8.0),
    // A8: Original direction, more extreme typing
    (1.0, 0.5, 5.0, 8.0, 0.5, 15.0),
    // A9: All balanced
    (2.5, 2.5, 2.5, 2.5, 1.5, 8.0),
];

/// Grid B: Tolerance configurations (6 total)
/// Format: (h_tol_init, h_tol_final, c_tol_init, c_tol_final)
const GRID_B: &[(f64, f64, f64, f64)] = &[
    // B1: Current defaults
    (0.05, 0.02, 3.0, 1.0),
    // B2: Looser
    (0.08, 0.03, 4.0, 1.5),
    // B3: Tighter
    (0.03, 0.01, 2.0, 0.5),
    // B4: Very loose
    (0.10, 0.05, 5.0, 2.0),
    // B5: Slightly tight
    (0.04, 0.02, 2.5, 0.8),
    // B6: Stay loose longer
    (0.06, 0.04, 3.5, 1.5),
];

/// Grid C: BP dynamics configurations (4 total)
/// Format: (damping, exploration_fraction, max_iterations)
const GRID_C: &[(f64, f64, usize)] = &[
    // C1: Current defaults
    (0.5, 0.3, 100),
    // C2: More damping, more iterations
    (0.7, 0.3, 150),
    // C3: Less damping, more exploration
    (0.3, 0.5, 100),
    // C4: More refinement time
    (0.6, 0.2, 200),
];

/// Grid search result for a single configuration
#[derive(Serialize)]
struct GridSearchResult {
    config_id: String,
    grid_a: usize,
    grid_b: usize,
    grid_c: usize,
    params: ParamSnapshot,
    stretches: Vec<StretchResult>,
    avg_accuracy: f64,
    avg_backbone_accuracy: f64,
    total_time_ms: f64,
}

/// Snapshot of parameter values for JSON output
#[derive(Serialize)]
struct ParamSnapshot {
    tocsy_init: f64,
    tocsy_final: f64,
    typing_init: f64,
    typing_final: f64,
    sequential: f64,
    seq_type: f64,
    h_tol_init: f64,
    h_tol_final: f64,
    c_tol_init: f64,
    c_tol_final: f64,
    damping: f64,
    explore_frac: f64,
    max_iter: usize,
}

/// Result for a single stretch
#[derive(Serialize)]
struct StretchResult {
    range: String,
    sequence: String,
    overall_accuracy: f64,
    backbone_accuracy: f64,
    timing_ms: f64,
    per_aa: HashMap<String, JsonAAStats>,
}

/// Run comprehensive grid search over parameter space
fn run_grid_search_mode(
    entry_id: u32,
    stretches: &str,
    output_dir: PathBuf,
    grid_a_filter: Option<usize>,
    grid_b_filter: Option<usize>,
    grid_c_filter: Option<usize>,
    verbose: bool,
    filter: &ExperimentFilter,
) {
    println!("=== GRID SEARCH ===");
    println!("BMRB Entry: {}", entry_id);
    println!("Stretches: {}", stretches);

    // Parse stretches
    let stretch_ranges: Vec<&str> = stretches.split(',').map(|s| s.trim()).collect();
    println!("Testing {} stretches: {:?}", stretch_ranges.len(), stretch_ranges);

    // Create output directory
    if let Err(e) = std::fs::create_dir_all(&output_dir) {
        eprintln!("Error creating output directory: {}", e);
        std::process::exit(1);
    }

    // Fetch all shifts once
    println!("\nFetching BMRB entry {}...", entry_id);
    let all_shifts = match fetch_entry_shifts(entry_id) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error fetching BMRB entry: {}", e);
            std::process::exit(1);
        }
    };

    let full_sequence = extract_sequence(&all_shifts);
    println!("Full sequence: {} ({} residues)", full_sequence, full_sequence.len());

    // Pre-filter data for each stretch to avoid repeated fetching
    let mut stretch_data: Vec<(String, Vec<nmraster_lib::testdata::DepositedShift>)> = Vec::new();
    for range in &stretch_ranges {
        let mut shifts = match filter_residue_range(&all_shifts, range) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error filtering stretch {}: {}", range, e);
                continue;
            }
        };
        renumber_shifts(&mut shifts);
        if !shifts.is_empty() {
            stretch_data.push((range.to_string(), shifts));
        }
    }

    if stretch_data.is_empty() {
        eprintln!("No valid stretches found");
        std::process::exit(1);
    }

    // Determine grid indices to run
    let a_indices: Vec<usize> = match grid_a_filter {
        Some(i) if i >= 1 && i <= GRID_A.len() => vec![i - 1],
        Some(i) => {
            eprintln!("Invalid grid_a index: {} (valid: 1-{})", i, GRID_A.len());
            std::process::exit(1);
        }
        None => (0..GRID_A.len()).collect(),
    };

    let b_indices: Vec<usize> = match grid_b_filter {
        Some(i) if i >= 1 && i <= GRID_B.len() => vec![i - 1],
        Some(i) => {
            eprintln!("Invalid grid_b index: {} (valid: 1-{})", i, GRID_B.len());
            std::process::exit(1);
        }
        None => (0..GRID_B.len()).collect(),
    };

    let c_indices: Vec<usize> = match grid_c_filter {
        Some(i) if i >= 1 && i <= GRID_C.len() => vec![i - 1],
        Some(i) => {
            eprintln!("Invalid grid_c index: {} (valid: 1-{})", i, GRID_C.len());
            std::process::exit(1);
        }
        None => (0..GRID_C.len()).collect(),
    };

    let total_configs = a_indices.len() * b_indices.len() * c_indices.len();
    let total_runs = total_configs * stretch_data.len();
    println!(
        "\nRunning {} configurations × {} stretches = {} total runs",
        total_configs,
        stretch_data.len(),
        total_runs
    );

    let mut all_results: Vec<GridSearchResult> = Vec::new();
    let mut config_num = 0;

    for &a_idx in &a_indices {
        for &b_idx in &b_indices {
            for &c_idx in &c_indices {
                config_num += 1;
                let config_id = format!("A{}B{}C{}", a_idx + 1, b_idx + 1, c_idx + 1);

                // Build params from grid values
                let (tocsy_init, tocsy_final, typing_init, typing_final, sequential, seq_type) = GRID_A[a_idx];
                let (h_tol_init, h_tol_final, c_tol_init, c_tol_final) = GRID_B[b_idx];
                let (damping, explore_frac, max_iter) = GRID_C[c_idx];

                let params = UnifiedAssignmentParams {
                    tocsy_weight_initial: tocsy_init,
                    tocsy_weight_final: tocsy_final,
                    typing_weight_initial: typing_init,
                    typing_weight_final: typing_final,
                    sequential_weight: sequential,
                    sequence_type_weight: seq_type,
                    h_tolerance_initial: h_tol_init,
                    h_tolerance_final: h_tol_final,
                    c_tolerance_initial: c_tol_init,
                    c_tolerance_final: c_tol_final,
                    damping,
                    exploration_fraction: explore_frac,
                    max_iterations: max_iter,
                    verbose,
                    ..Default::default()
                };

                print!("\n[{}/{}] Config {} ", config_num, total_configs, config_id);
                print!("(tocsy {:.1}/{:.1}, typing {:.1}/{:.1}, seq {:.1}, seqtype {:.1}, ",
                    tocsy_init, tocsy_final, typing_init, typing_final, sequential, seq_type);
                println!("htol {:.3}/{:.3}, ctol {:.1}/{:.1}, damp {:.1}, exp {:.1}, iter {})",
                    h_tol_init, h_tol_final, c_tol_init, c_tol_final, damping, explore_frac, max_iter);

                let mut stretch_results: Vec<StretchResult> = Vec::new();
                let mut total_time = 0.0;

                for (range, shifts) in &stretch_data {
                    let sequence = extract_sequence(shifts);
                    let (hsqc_15n, hsqc_13c, tocsy_peaks, noesy, hsqc_tocsy_15n, hsqc_tocsy_13c, hsqc_tocsy_15n_3d, hsqc_tocsy_13c_3d, hnco, hnca, hncacb, cbcaconh, hbhaconh, ground_truth) =
                        generate_peaks_from_bmrb(shifts, filter);

                    let start = Instant::now();
                    let results = run_unified_assignment(
                        &hsqc_15n, &hsqc_13c, &tocsy_peaks, &noesy, &hsqc_tocsy_15n, &hsqc_tocsy_13c,
                        &hsqc_tocsy_15n_3d, &hsqc_tocsy_13c_3d,
                        &hnco, &hnca, &hncacb, &cbcaconh, &hbhaconh,
                        &sequence, &params
                    );
                    let elapsed = start.elapsed();
                    let timing_ms = elapsed.as_secs_f64() * 1000.0;
                    total_time += timing_ms;

                    let summary = evaluate_results(&results, &ground_truth, &sequence);

                    let overall_acc = if summary.total_peaks > 0 {
                        summary.correct as f64 / summary.total_peaks as f64
                    } else {
                        0.0
                    };

                    let backbone_acc = if summary.backbone_peaks > 0 {
                        summary.backbone_correct as f64 / summary.backbone_peaks as f64
                    } else {
                        0.0
                    };

                    // Convert per-AA stats
                    let mut per_aa = HashMap::new();
                    for (&aa, stats) in &summary.per_aa {
                        let accuracy = if stats.total > 0 {
                            stats.correct as f64 / stats.total as f64
                        } else {
                            0.0
                        };
                        per_aa.insert(
                            aa.to_string(),
                            JsonAAStats {
                                total: stats.total,
                                correct: stats.correct,
                                accuracy,
                            },
                        );
                    }

                    print!("  {}: {:.1}% ", range, overall_acc * 100.0);

                    stretch_results.push(StretchResult {
                        range: range.clone(),
                        sequence: sequence.clone(),
                        overall_accuracy: overall_acc,
                        backbone_accuracy: backbone_acc,
                        timing_ms,
                        per_aa,
                    });
                }

                let avg_accuracy = stretch_results.iter().map(|r| r.overall_accuracy).sum::<f64>()
                    / stretch_results.len() as f64;
                let avg_backbone = stretch_results.iter().map(|r| r.backbone_accuracy).sum::<f64>()
                    / stretch_results.len() as f64;

                println!("-> AVG: {:.1}%, BB: {:.1}%", avg_accuracy * 100.0, avg_backbone * 100.0);

                let result = GridSearchResult {
                    config_id: config_id.clone(),
                    grid_a: a_idx + 1,
                    grid_b: b_idx + 1,
                    grid_c: c_idx + 1,
                    params: ParamSnapshot {
                        tocsy_init,
                        tocsy_final,
                        typing_init,
                        typing_final,
                        sequential,
                        seq_type,
                        h_tol_init,
                        h_tol_final,
                        c_tol_init,
                        c_tol_final,
                        damping,
                        explore_frac,
                        max_iter,
                    },
                    stretches: stretch_results,
                    avg_accuracy,
                    avg_backbone_accuracy: avg_backbone,
                    total_time_ms: total_time,
                };

                // Write individual result file
                let json_path = output_dir.join(format!("{}.json", config_id));
                if let Ok(json_str) = serde_json::to_string_pretty(&result) {
                    let _ = std::fs::write(&json_path, json_str);
                }

                all_results.push(result);
            }
        }
    }

    // Sort results by average accuracy
    all_results.sort_by(|a, b| b.avg_accuracy.partial_cmp(&a.avg_accuracy).unwrap());

    // Print summary
    println!("\n{}", "=".repeat(79));
    println!("{:^79}", "GRID SEARCH RESULTS");
    println!("{}", "=".repeat(79));
    println!("\nTop 10 configurations:");
    println!("{:<10} {:>8} {:>8} {:>8}", "Config", "Avg%", "BB%", "Time(s)");
    println!("{}", "-".repeat(40));

    for (i, result) in all_results.iter().take(10).enumerate() {
        println!(
            "{:<10} {:>7.1}% {:>7.1}% {:>7.1}",
            result.config_id,
            result.avg_accuracy * 100.0,
            result.avg_backbone_accuracy * 100.0,
            result.total_time_ms / 1000.0
        );
    }

    // Write summary file
    let summary_path = output_dir.join("summary.json");
    if let Ok(json_str) = serde_json::to_string_pretty(&all_results) {
        let _ = std::fs::write(&summary_path, json_str);
    }

    println!("\nResults written to: {}", output_dir.display());
    println!("Summary: {}", summary_path.display());

    // Print best config details
    if let Some(best) = all_results.first() {
        println!("\n{}", "=".repeat(79));
        println!("BEST CONFIGURATION: {}", best.config_id);
        println!("{}", "=".repeat(79));
        println!("Parameters:");
        println!("  --tocsy-init={:.2} --tocsy-final={:.2}", best.params.tocsy_init, best.params.tocsy_final);
        println!("  --typing-init={:.2} --typing-final={:.2}", best.params.typing_init, best.params.typing_final);
        println!("  --sequential={:.2} --seq-type={:.2}", best.params.sequential, best.params.seq_type);
        println!("  --h-tol-init={:.3} --h-tol-final={:.3}", best.params.h_tol_init, best.params.h_tol_final);
        println!("  --c-tol-init={:.2} --c-tol-final={:.2}", best.params.c_tol_init, best.params.c_tol_final);
        println!("  --damping={:.2} --explore-frac={:.2} --max-iter={}",
            best.params.damping, best.params.explore_frac, best.params.max_iter);
        println!("\nPer-stretch breakdown:");
        for stretch in &best.stretches {
            println!("  {}: {:.1}% overall, {:.1}% backbone",
                stretch.range, stretch.overall_accuracy * 100.0, stretch.backbone_accuracy * 100.0);
        }
    }
}
