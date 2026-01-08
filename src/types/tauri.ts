/**
 * TypeScript types matching Rust structs from Tauri commands.
 * These types ensure type safety when invoking backend commands.
 */

// ============================================================================
// Spectrum Types
// ============================================================================

/** Response from get_spectrum_1d command */
export interface SpectrumDataResponse {
  id: string;
  name: string;
  experiment_type: string;
  real: number[];
  imag: number[] | null;
  ppm_axis: number[];
  is_processed: boolean;
}

/** Request for load_spectrum_1d command */
export interface LoadSpectrumRequest {
  name: string;
  experiment_type: string;
  real: number[];
  imag?: number[];
  spectral_width_hz: number;
  carrier_offset_ppm: number;
  spectrometer_frequency_mhz: number;
}

/** Spectrum summary from list_spectra command */
export interface SpectrumInfo {
  id: string;
  name: string;
  dimensions: number;
  experiment_type: string;
  num_points: number;
  is_processed: boolean;
}

/** Basic peak info from get_spectrum_peaks command */
export interface PeakInfo {
  id: string;
  position_ppm: number[];
  intensity: number;
  volume: number | null;
  annotation: string | null;
}

/** Response from get_spectrum_2d or generate_demo_2d commands */
export interface Spectrum2DDataResponse {
  id: string;
  /** 2D intensity data as nested arrays [F1][F2] */
  data: number[][];
  /** PPM axis for F1 (indirect dimension, e.g., 15N or 13C) */
  ppm_axis_f1: number[];
  /** PPM axis for F2 (direct dimension, typically 1H) */
  ppm_axis_f2: number[];
  /** Experiment type (e.g., "HSQC", "TOCSY", "NOESY") */
  experiment_type: string;
  /** Estimated noise floor for contour level calculation */
  noise_floor: number;
  /** Number of peaks (only populated by generate_demo_2d) */
  num_peaks: number;
}

// ============================================================================
// Assignment Types
// ============================================================================

/** Molecule info from get_active_molecule command */
export interface MoleculeInfo {
  id: string;
  name: string;
  sequence: string;
  num_residues: number;
  num_atoms: number;
}

/** Residue info from get_molecule_residues command */
export interface ResidueInfo {
  id: string;
  sequence_code: number;
  residue_name: string;
  one_letter_code: string;
  chain_code: string;
}

/** Atom info from get_residue_atoms command */
export interface AtomInfo {
  id: string;
  atom_name: string;
  element: string;
}

/** Chemical shift info from get_shifts_for_residue command */
export interface ShiftInfo {
  id: string;
  atom_name: string;
  value: number;
  error: number | null;
  confidence: number;
}

/** Shift list info from list_shift_lists command */
export interface ShiftListInfo {
  id: string;
  name: string;
  molecule_id: string;
  num_shifts: number;
}

/** Result from global assignment */
export interface AssignmentResult {
  atom_id: string;
  atom_name: string;
  residue_seq_code: number;
  residue_name: string;
  assigned_shift: number;
  confidence: number;
  alternative_shifts: [number, number][]; // [shift, probability]
}

/** Parameters for run_assignment command */
export interface AssignmentParams {
  max_iterations?: number;
  tolerance?: number;
  include_sidechain?: boolean;
}

/** Confidence level for display */
export type ConfidenceLevel = 'high' | 'medium' | 'low';

/** Spin system info for a residue (frontend-only, derived from AssignmentResult) */
export interface SpinSystemInfo {
  residue_seq_code: number;
  residue_name: string;
  atoms: SpinSystemAtom[];
  correlations: SpinSystemCorrelation[];
  overall_confidence: number;
}

/** Atom within a spin system */
export interface SpinSystemAtom {
  atom_id: string;
  atom_name: string;
  assigned_shift: number | null;
  confidence: number;
  alternatives: [number, number][]; // [shift, probability]
}

/** Correlation between atoms in a spin system */
export interface SpinSystemCorrelation {
  from_atom: string;
  to_atom: string;
  correlation_type: 'scalar' | 'noe' | 'sequential';
  strength: number;
}

// ============================================================================
// Analysis Types
// ============================================================================

/** Parameters for pick_peaks_1d command */
export interface PeakPickingParams {
  min_snr: number;
  min_intensity?: number;
  noise_region?: [number, number]; // [start_ppm, end_ppm]
}

/** Result from pick_peaks_1d command */
export interface PickedPeak {
  id: string;
  ppm: number;
  intensity: number;
  snr: number;
}

/** Result from integrate_peak command */
export interface IntegrationResult {
  center_ppm: number;
  width_ppm: number;
  volume: number;
  max_intensity: number;
  num_points: number;
}

// ============================================================================
// Database Types
// ============================================================================

/** Molecule info from database */
export interface DbMoleculeInfo {
  id: string;
  name: string;
}

/** Database statistics */
export interface DbStats {
  molecules: number;
  spectra: number;
  peaks: number;
  chemical_shifts: number;
  distance_constraints: number;
}

/** Application state statistics */
export interface StateStats {
  molecules: number;
  spectra_1d: number;
  spectra_2d: number;
  peak_lists: number;
  shift_lists: number;
  constraint_sets: number;
}

// ============================================================================
// UI State Types (Frontend-only)
// ============================================================================

/** View bounds for spectrum display */
export interface ViewBounds {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

/** Mouse position in spectrum coordinates */
export interface SpectrumCoords {
  ppm: number;
  intensity: number;
}

/** Tool mode for spectrum interaction */
export type ToolMode = 'select' | 'zoom' | 'pan' | 'peak-pick' | 'integrate';

/** Assignment status for residue coloring */
export type AssignmentStatus = 'unassigned' | 'partial' | 'complete';

// ============================================================================
// Test Data Types (BMRB Statistics)
// ============================================================================

/** BMRB chemical shift statistics for an atom */
export interface AtomShiftStatistics {
  residue_name: string;
  atom_name: string;
  nucleus: string;
  mean: number;
  std_dev: number;
  min: number;
  max: number;
  count: number;
}

/** Assignment of a peak to a specific atom */
export interface AtomAssignment {
  residue_seq_code: number;
  residue_name: string;
  atom_name: string;
}

/** Ground truth peak with known position and assignment */
export interface GroundTruthPeak {
  position_ppm: number[];
  intensity: number;
  assignments: AtomAssignment[];
  line_width_hz: number | null;
}

/** Collection of ground truth peaks for a spectrum */
export interface GroundTruthPeaks {
  molecule_name: string;
  sequence: string;
  experiment_type: string;
  dimensions: number;
  axis_labels: string[];
  spectral_widths_ppm: number[];
  carrier_offsets_ppm: number[];
  peaks: GroundTruthPeak[];
}

/** Parameters for test data generation */
export interface TestDataParams {
  spectrometer_frequency_mhz?: number;
  num_points_1d?: number;
  line_width_hz?: number;
}

/** Summary of generated test data */
export interface TestDataSummary {
  sequence_length: number;
  num_residues: number;
  experiments_generated: string[];
  total_ground_truth_peaks: number;
}

/** Result of test data generation */
export interface TestDataResult {
  molecule_id: string;
  spectra_ids: Record<string, string>;
  ground_truth: Record<string, GroundTruthPeaks>;
  summary: TestDataSummary;
}

// ============================================================================
// Import Types
// ============================================================================

/** Request for importing a Bruker spectrum */
export interface ImportBrukerRequest {
  /** Path to the Bruker experiment directory */
  path: string;
  /** Import processed data (true) or FID (false) */
  processed: boolean;
  /** Processing number (default: 1) */
  procno?: number;
  /** Optional name override */
  name?: string;
}

/** Request for importing an NMRPipe spectrum */
export interface ImportNmrPipeRequest {
  /** Path to the NMRPipe file */
  path: string;
  /** Optional name override */
  name?: string;
}

/** Result of a successful import */
export interface ImportResult {
  /** UUID of the imported spectrum */
  spectrum_id: string;
  /** Name of the spectrum */
  name: string;
  /** Number of dimensions (1, 2, 3, or 4) */
  dimensions: number;
  /** Experiment type (e.g., "Proton1D", "HSQC") */
  experiment_type: string;
  /** Number of points in each dimension */
  num_points: number[];
  /** Whether the spectrum is processed (vs FID) */
  is_processed: boolean;
  /** Source format ("Bruker" or "NMRPipe") */
  source_format: string;
}
