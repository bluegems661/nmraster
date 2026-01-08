/**
 * Zustand store for molecule and chemical shift assignment state.
 */

import { create } from 'zustand';
import type {
  MoleculeInfo,
  ResidueInfo,
  ShiftInfo,
  ShiftListInfo,
  AssignmentStatus,
  AssignmentResult,
  AssignmentParams,
  SpinSystemInfo,
  SpinSystemAtom,
  SpinSystemCorrelation,
  ConfidenceLevel,
} from '../types/tauri';
import * as api from '../lib/tauri';

interface AssignmentState {
  // Data
  molecule: MoleculeInfo | null;
  residues: ResidueInfo[];
  shiftLists: ShiftListInfo[];
  shiftsPerResidue: Map<number, ShiftInfo[]>; // keyed by sequence_code

  // Assignment results
  assignmentResults: AssignmentResult[];
  spinSystems: Map<number, SpinSystemInfo>; // keyed by residue seq_code

  // Selection
  activeShiftListId: string | null;
  selectedResidueSeqCode: number | null;
  selectedAtomId: string | null;

  // Loading state
  isLoading: boolean;
  isAssigning: boolean;
  error: string | null;

  // Confidence thresholds
  confidenceThresholds: {
    high: number;
    medium: number;
  };

  // Actions - Molecule
  loadMolecule: (name: string, sequence: string, chainCode?: string) => Promise<void>;
  fetchActiveMolecule: () => Promise<void>;
  fetchResidues: () => Promise<void>;

  // Actions - Shift Lists
  fetchShiftLists: () => Promise<void>;
  createShiftList: (name: string) => Promise<string | null>;
  setActiveShiftList: (id: string | null) => void;

  // Actions - Shifts
  fetchShiftsForResidue: (seqCode: number) => Promise<void>;

  // Actions - Selection
  selectResidue: (seqCode: number | null) => void;
  selectAtom: (atomId: string | null) => void;

  // Actions - Assignment
  runGlobalAssignment: (params?: AssignmentParams) => Promise<void>;
  buildSpinSystems: () => void;
  clearAssignmentResults: () => void;

  // Computed helpers
  getAssignmentStatus: (seqCode: number) => AssignmentStatus;
  getConfidenceLevel: (confidence: number) => ConfidenceLevel;
}

export const useAssignmentStore = create<AssignmentState>((set, get) => ({
  // Initial state
  molecule: null,
  residues: [],
  shiftLists: [],
  shiftsPerResidue: new Map(),
  assignmentResults: [],
  spinSystems: new Map(),
  activeShiftListId: null,
  selectedResidueSeqCode: null,
  selectedAtomId: null,
  isLoading: false,
  isAssigning: false,
  error: null,
  confidenceThresholds: {
    high: 0.8,
    medium: 0.5,
  },

  // Load a molecule from sequence
  loadMolecule: async (name: string, sequence: string, chainCode?: string) => {
    set({ isLoading: true, error: null });
    try {
      await api.loadMoleculeFromSequence(name, sequence, chainCode);
      await get().fetchActiveMolecule();
      await get().fetchResidues();
      set({ isLoading: false });
    } catch (err) {
      set({ error: String(err), isLoading: false });
    }
  },

  // Fetch active molecule
  fetchActiveMolecule: async () => {
    try {
      const molecule = await api.getActiveMolecule();
      console.log('[Store] Fetched molecule from API:', molecule);
      console.log('[Store] molecule.num_residues:', molecule?.num_residues);
      set({ molecule });
      // Verify state was updated
      const newState = get();
      console.log('[Store] After set(), store molecule:', newState.molecule);
      console.log('[Store] After set(), num_residues:', newState.molecule?.num_residues);
    } catch (err) {
      console.error('fetchActiveMolecule error:', err);
      set({ error: String(err) });
    }
  },

  // Fetch residues for active molecule
  fetchResidues: async () => {
    const { molecule } = get();
    if (!molecule) return;

    try {
      const residues = await api.getMoleculeResidues(molecule.id);
      set({ residues });
    } catch (err) {
      set({ error: String(err) });
    }
  },

  // Fetch all shift lists
  fetchShiftLists: async () => {
    try {
      const shiftLists = await api.listShiftLists();
      set({ shiftLists });
    } catch (err) {
      set({ error: String(err) });
    }
  },

  // Create a new shift list
  createShiftList: async (name: string) => {
    const { molecule } = get();
    if (!molecule) return null;

    set({ isLoading: true, error: null });
    try {
      const id = await api.createShiftList(name, molecule.id);
      await get().fetchShiftLists();
      set({ activeShiftListId: id, isLoading: false });
      return id;
    } catch (err) {
      set({ error: String(err), isLoading: false });
      return null;
    }
  },

  // Set active shift list
  setActiveShiftList: (id: string | null) => {
    set({ activeShiftListId: id, shiftsPerResidue: new Map() });
  },

  // Fetch shifts for a residue
  fetchShiftsForResidue: async (seqCode: number) => {
    const { activeShiftListId } = get();
    if (!activeShiftListId) return;

    try {
      const shifts = await api.getShiftsForResidue(activeShiftListId, seqCode);
      const shiftsPerResidue = new Map(get().shiftsPerResidue);
      shiftsPerResidue.set(seqCode, shifts);
      set({ shiftsPerResidue });
    } catch (err) {
      set({ error: String(err) });
    }
  },

  // Select a residue
  selectResidue: (seqCode: number | null) => {
    set({ selectedResidueSeqCode: seqCode, selectedAtomId: null });
    if (seqCode !== null) {
      get().fetchShiftsForResidue(seqCode);
    }
  },

  // Select an atom
  selectAtom: (atomId: string | null) => {
    set({ selectedAtomId: atomId });

    // If selecting an atom, also select its residue
    if (atomId) {
      const result = get().assignmentResults.find((r) => r.atom_id === atomId);
      if (result) {
        set({ selectedResidueSeqCode: result.residue_seq_code });
      }
    }
  },

  // Run global assignment
  runGlobalAssignment: async (params?: AssignmentParams) => {
    const { molecule, activeShiftListId } = get();
    if (!molecule) {
      console.error('No molecule loaded');
      set({ error: 'No molecule loaded' });
      return;
    }

    console.log('Running assignment for molecule:', molecule.id);
    set({ isAssigning: true, error: null });
    try {
      const results = await api.runAssignment(
        molecule.id,
        activeShiftListId ?? undefined,
        params
      );
      console.log('Assignment results:', results);
      set({
        assignmentResults: results,
        isAssigning: false,
      });
      get().buildSpinSystems();
    } catch (err) {
      console.error('Assignment failed:', err);
      set({ error: String(err), isAssigning: false });
    }
  },

  // Build spin systems from assignment results
  buildSpinSystems: () => {
    const { assignmentResults, residues } = get();
    const spinSystems = new Map<number, SpinSystemInfo>();

    // Group results by residue
    for (const residue of residues) {
      const residueAtoms = assignmentResults.filter(
        (r) => r.residue_seq_code === residue.sequence_code
      );

      const atoms: SpinSystemAtom[] = residueAtoms.map((a) => ({
        atom_id: a.atom_id,
        atom_name: a.atom_name,
        assigned_shift: a.assigned_shift,
        confidence: a.confidence,
        alternatives: a.alternative_shifts,
      }));

      // Build correlations (backbone connectivity)
      const correlations: SpinSystemCorrelation[] = [];

      // H-N scalar coupling
      const hasH = atoms.some((a) => a.atom_name === 'H');
      const hasN = atoms.some((a) => a.atom_name === 'N');
      if (hasH && hasN) {
        correlations.push({
          from_atom: 'H',
          to_atom: 'N',
          correlation_type: 'scalar',
          strength: 1.0,
        });
      }

      // N-CA connectivity
      const hasCA = atoms.some((a) => a.atom_name === 'CA');
      if (hasN && hasCA) {
        correlations.push({
          from_atom: 'N',
          to_atom: 'CA',
          correlation_type: 'scalar',
          strength: 1.0,
        });
      }

      // CA-C connectivity
      const hasC = atoms.some((a) => a.atom_name === 'C');
      if (hasCA && hasC) {
        correlations.push({
          from_atom: 'CA',
          to_atom: 'C',
          correlation_type: 'scalar',
          strength: 1.0,
        });
      }

      // CA-CB connectivity
      const hasCB = atoms.some((a) => a.atom_name === 'CB');
      if (hasCA && hasCB) {
        correlations.push({
          from_atom: 'CA',
          to_atom: 'CB',
          correlation_type: 'scalar',
          strength: 1.0,
        });
      }

      // CA-HA connectivity
      const hasHA = atoms.some((a) => a.atom_name === 'HA');
      if (hasCA && hasHA) {
        correlations.push({
          from_atom: 'CA',
          to_atom: 'HA',
          correlation_type: 'scalar',
          strength: 1.0,
        });
      }

      const overall =
        atoms.length > 0
          ? atoms.reduce((sum, a) => sum + a.confidence, 0) / atoms.length
          : 0;

      spinSystems.set(residue.sequence_code, {
        residue_seq_code: residue.sequence_code,
        residue_name: residue.residue_name,
        atoms,
        correlations,
        overall_confidence: overall,
      });
    }

    set({ spinSystems });
  },

  // Clear assignment results
  clearAssignmentResults: () => {
    set({
      assignmentResults: [],
      spinSystems: new Map(),
    });
  },

  // Get assignment status for a residue
  getAssignmentStatus: (seqCode: number): AssignmentStatus => {
    const { spinSystems } = get();
    const spinSystem = spinSystems.get(seqCode);

    if (!spinSystem || spinSystem.atoms.length === 0) {
      // Fall back to shifts
      const shifts = get().shiftsPerResidue.get(seqCode);
      if (!shifts || shifts.length === 0) return 'unassigned';

      const backboneAtoms = ['N', 'H', 'CA', 'C'];
      const assignedBackbone = shifts.filter((s) =>
        backboneAtoms.includes(s.atom_name)
      ).length;

      if (assignedBackbone >= 4) return 'complete';
      if (assignedBackbone > 0) return 'partial';
      return 'unassigned';
    }

    // Use spin system data
    const backboneAtoms = ['N', 'H', 'CA', 'C'];
    const assignedBackbone = spinSystem.atoms.filter(
      (a) => backboneAtoms.includes(a.atom_name) && a.assigned_shift !== null
    ).length;

    if (assignedBackbone >= 4) return 'complete';
    if (assignedBackbone > 0) return 'partial';
    return 'unassigned';
  },

  // Get confidence level
  getConfidenceLevel: (confidence: number): ConfidenceLevel => {
    const { confidenceThresholds } = get();
    if (confidence >= confidenceThresholds.high) return 'high';
    if (confidence >= confidenceThresholds.medium) return 'medium';
    return 'low';
  },
}));
