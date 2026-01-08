/**
 * Type-safe wrappers for Tauri commands.
 * All backend communication goes through these functions.
 */

import { invoke } from '@tauri-apps/api/core';
import type {
  SpectrumDataResponse,
  LoadSpectrumRequest,
  SpectrumInfo,
  PeakInfo,
  MoleculeInfo,
  ResidueInfo,
  AtomInfo,
  ShiftInfo,
  ShiftListInfo,
  AssignmentResult,
  AssignmentParams,
  PeakPickingParams,
  PickedPeak,
  IntegrationResult,
  DbMoleculeInfo,
  DbStats,
  StateStats,
} from '../types/tauri';

// Re-export types for convenience
export type { GroundTruthPeaks, TestDataResult, AtomShiftStatistics } from '../types/tauri';

// ============================================================================
// Spectrum Commands
// ============================================================================

/** Load a 1D spectrum from array data */
export async function loadSpectrum1D(request: LoadSpectrumRequest): Promise<string> {
  return invoke<string>('load_spectrum_1d', { request });
}

/** Get spectrum data by ID */
export async function getSpectrum1D(id: string): Promise<SpectrumDataResponse | null> {
  return invoke<SpectrumDataResponse | null>('get_spectrum_1d', { id });
}

/** Process a 1D FID spectrum (apply FFT) */
export async function processSpectrum1D(
  id: string,
  zeroFillFactor?: number
): Promise<string> {
  return invoke<string>('process_spectrum_1d', {
    id,
    zero_fill_factor: zeroFillFactor,
  });
}

/** List all loaded spectra */
export async function listSpectra(): Promise<SpectrumInfo[]> {
  return invoke<SpectrumInfo[]>('list_spectra');
}

/** Get peaks for a spectrum */
export async function getSpectrumPeaks(spectrumId: string): Promise<PeakInfo[]> {
  return invoke<PeakInfo[]>('get_spectrum_peaks', { spectrumId });
}

// ============================================================================
// Assignment Commands
// ============================================================================

/** Load a molecule from sequence */
export async function loadMoleculeFromSequence(
  name: string,
  sequence: string,
  chainCode?: string
): Promise<string> {
  return invoke<string>('load_molecule_from_sequence', {
    name,
    sequence,
    chain_code: chainCode,
  });
}

/** Get active molecule */
export async function getActiveMolecule(): Promise<MoleculeInfo | null> {
  return invoke<MoleculeInfo | null>('get_active_molecule');
}

/** Get residues for a molecule */
export async function getMoleculeResidues(moleculeId: string): Promise<ResidueInfo[]> {
  return invoke<ResidueInfo[]>('get_molecule_residues', { moleculeId });
}

/** Get atoms for a residue */
export async function getResidueAtoms(
  moleculeId: string,
  residueSeqCode: number
): Promise<AtomInfo[]> {
  return invoke<AtomInfo[]>('get_residue_atoms', {
    moleculeId,
    residueSeqCode,
  });
}

/** Create a new chemical shift list */
export async function createShiftList(name: string, moleculeId: string): Promise<string> {
  return invoke<string>('create_shift_list', { name, moleculeId });
}

/** Add a chemical shift to a list */
export async function addChemicalShift(
  listId: string,
  atomId: string,
  atomName: string,
  residueSeqCode: number,
  residueName: string,
  chainCode: string,
  value: number,
  nucleus: string
): Promise<string> {
  return invoke<string>('add_chemical_shift', {
    list_id: listId,
    atom_id: atomId,
    atom_name: atomName,
    residue_seq_code: residueSeqCode,
    residue_name: residueName,
    chain_code: chainCode,
    value,
    nucleus,
  });
}

/** Get chemical shifts for a residue */
export async function getShiftsForResidue(
  listId: string,
  residueSeqCode: number
): Promise<ShiftInfo[]> {
  return invoke<ShiftInfo[]>('get_shifts_for_residue', {
    list_id: listId,
    residue_seq_code: residueSeqCode,
  });
}

/** List all chemical shift lists */
export async function listShiftLists(): Promise<ShiftListInfo[]> {
  return invoke<ShiftListInfo[]>('list_shift_lists');
}

/** Run global assignment using factor graph and belief propagation */
export async function runAssignment(
  moleculeId: string,
  peakListId?: string,
  params?: AssignmentParams
): Promise<AssignmentResult[]> {
  return invoke<AssignmentResult[]>('run_assignment', {
    moleculeId,
    peakListId,
    params,
  });
}

// ============================================================================
// Analysis Commands
// ============================================================================

/** Pick peaks in a 1D spectrum */
export async function pickPeaks1D(
  spectrumId: string,
  params: PeakPickingParams
): Promise<PickedPeak[]> {
  return invoke<PickedPeak[]>('pick_peaks_1d', {
    spectrumId,
    params,
  });
}

/** Integrate a peak */
export async function integratePeak(
  spectrumId: string,
  centerPpm: number,
  widthPpm: number
): Promise<IntegrationResult> {
  return invoke<IntegrationResult>('integrate_peak', {
    spectrumId,
    centerPpm,
    widthPpm,
  });
}

/** Clear all peaks for a spectrum */
export async function clearSpectrumPeaks(spectrumId: string): Promise<void> {
  return invoke<void>('clear_spectrum_peaks', { spectrumId });
}

// ============================================================================
// Database Commands
// ============================================================================

/** Initialize database at a path */
export async function initDatabase(path: string): Promise<void> {
  return invoke<void>('init_database', { path });
}

/** Save a molecule to the database */
export async function saveMoleculeToDb(moleculeId: string): Promise<void> {
  return invoke<void>('save_molecule_to_db', { moleculeId });
}

/** Load a molecule from the database */
export async function loadMoleculeFromDb(moleculeId: string): Promise<string | null> {
  return invoke<string | null>('load_molecule_from_db', { moleculeId });
}

/** List all molecules in the database */
export async function listMoleculesInDb(): Promise<DbMoleculeInfo[]> {
  return invoke<DbMoleculeInfo[]>('list_molecules_in_db');
}

/** Delete a molecule from the database */
export async function deleteMoleculeFromDb(moleculeId: string): Promise<void> {
  return invoke<void>('delete_molecule_from_db', { moleculeId });
}

/** Get database statistics */
export async function getDbStats(): Promise<DbStats> {
  return invoke<DbStats>('get_db_stats');
}

/** Get application state statistics */
export async function getAppStats(): Promise<StateStats> {
  return invoke<StateStats>('get_app_stats');
}

// ============================================================================
// Test Data Commands (BMRB-based)
// ============================================================================

import type {
  TestDataParams,
  TestDataResult,
  AtomShiftStatistics,
  GroundTruthPeaks,
  Spectrum2DDataResponse,
} from '../types/tauri';

/** Generate realistic test data from BMRB statistics */
export async function generateTestData(
  sequence: string,
  name: string,
  experimentTypes: string[],
  params?: TestDataParams
): Promise<TestDataResult> {
  return invoke<TestDataResult>('generate_test_data', {
    sequence,
    name,
    experimentTypes,
    params,
  });
}

/** Get BMRB statistics for a specific residue type */
export async function getBmrbStats(residueName: string): Promise<AtomShiftStatistics[]> {
  return invoke<AtomShiftStatistics[]>('get_bmrb_stats', { residueName });
}

/** Get BMRB statistics for a specific residue and atom */
export async function getBmrbShift(
  residueName: string,
  atomName: string
): Promise<AtomShiftStatistics | null> {
  return invoke<AtomShiftStatistics | null>('get_bmrb_shift', { residueName, atomName });
}

/** Get expected peak positions for the active molecule */
export async function getExpectedPeaks(experimentType: string): Promise<GroundTruthPeaks> {
  return invoke<GroundTruthPeaks>('get_expected_peaks', { experimentType });
}

/** Create a shift list from ground truth peaks for assignment testing */
export async function createShiftListFromGroundTruth(
  groundTruth: GroundTruthPeaks,
  listName: string
): Promise<string> {
  return invoke<string>('create_shift_list_from_ground_truth', {
    groundTruth,
    listName,
  });
}

// ============================================================================
// 2D Spectrum Commands
// ============================================================================

/**
 * Generate a demo 2D spectrum for the active molecule.
 * Supported types: HSQC, CHSQC, TOCSY, NOESY
 */
export async function generateDemo2D(
  experimentType: string
): Promise<Spectrum2DDataResponse> {
  return invoke<Spectrum2DDataResponse>('generate_demo_2d', {
    experimentType,
  });
}

/** Get a previously generated 2D spectrum by ID */
export async function getSpectrum2D(spectrumId: string): Promise<Spectrum2DDataResponse> {
  return invoke<Spectrum2DDataResponse>('get_spectrum_2d', {
    spectrumId,
  });
}

// Re-export for convenience
export type { Spectrum2DDataResponse } from '../types/tauri';

// ============================================================================
// Import Commands
// ============================================================================

import type {
  ImportBrukerRequest,
  ImportNmrPipeRequest,
  ImportResult,
} from '../types/tauri';

export type { ImportBrukerRequest, ImportNmrPipeRequest, ImportResult } from '../types/tauri';

/** Import a 1D Bruker spectrum */
export async function importBruker1D(request: ImportBrukerRequest): Promise<ImportResult> {
  return invoke<ImportResult>('import_bruker_1d', { request });
}

/** Import a 2D Bruker spectrum */
export async function importBruker2D(request: ImportBrukerRequest): Promise<ImportResult> {
  return invoke<ImportResult>('import_bruker_2d', { request });
}

/** Import a 1D NMRPipe spectrum */
export async function importNmrPipe1D(request: ImportNmrPipeRequest): Promise<ImportResult> {
  return invoke<ImportResult>('import_nmrpipe_1d', { request });
}

/** Import a 2D NMRPipe spectrum */
export async function importNmrPipe2D(request: ImportNmrPipeRequest): Promise<ImportResult> {
  return invoke<ImportResult>('import_nmrpipe_2d', { request });
}

/** Auto-detect file format and import */
export async function importSpectrumAuto(path: string, name?: string): Promise<ImportResult> {
  return invoke<ImportResult>('import_spectrum_auto', { path, name });
}

/** Browse for a spectrum and import it using the file dialog */
export async function browseAndImportSpectrum(): Promise<ImportResult | null> {
  const { open } = await import('@tauri-apps/plugin-dialog');

  const selected = await open({
    directory: false,
    multiple: false,
    title: 'Select NMR Spectrum',
    filters: [
      { name: 'NMRPipe', extensions: ['ft1', 'ft2', 'fid'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  });

  if (selected && typeof selected === 'string') {
    return importSpectrumAuto(selected);
  }

  return null;
}

/** Browse for a Bruker experiment directory and import it */
export async function browseAndImportBruker(): Promise<ImportResult | null> {
  const { open } = await import('@tauri-apps/plugin-dialog');

  const selected = await open({
    directory: true,
    multiple: false,
    title: 'Select Bruker Experiment Directory',
  });

  if (selected && typeof selected === 'string') {
    return importSpectrumAuto(selected);
  }

  return null;
}
