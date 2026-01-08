/**
 * Main application layout for NMRaster.
 */

import { useCallback, useEffect, useState } from 'react';
import { SpectrumViewer } from './components/spectrum/SpectrumViewer';
import { SpectrumViewer2D } from './components/spectrum/SpectrumViewer2D';
import { SpectrumList } from './components/spectrum/SpectrumList';
import { PeakList } from './components/spectrum/PeakList';
import type { Spectrum2DDataResponse } from './types/tauri';
import { SequenceViewer } from './components/molecule/SequenceViewer';
import { AssignmentTable, SpinSystemView } from './components/assignment';
import { useSpectrumStore } from './stores/spectrumStore';
import { useAssignmentStore } from './stores/assignmentStore';
import { useUIStore } from './stores/uiStore';
import * as api from './lib/tauri';
import { browseAndImportSpectrum, browseAndImportBruker } from './lib/tauri';

function App() {
  const { fetchSpectraList, setActiveSpectrum } = useSpectrumStore();
  const {
    fetchActiveMolecule,
    runGlobalAssignment,
    isAssigning,
    molecule,
    assignmentResults,
    setActiveShiftList,
  } = useAssignmentStore();
  const {
    showSpectrumList,
    showPeakList,
    showSequence,
    showSpinSystemView,
    rightPanelTab,
    toggleSpectrumList,
    togglePeakList,
    toggleSequence,
    toggleSpinSystemView,
    setRightPanelTab,
  } = useUIStore();

  const [isLoadingDemo, setIsLoadingDemo] = useState(false);
  // Ground truth peaks stored for validation (used for comparing picked peaks)
  const [_groundTruth, setGroundTruth] = useState<Record<string, api.GroundTruthPeaks> | null>(null);

  // 2D spectrum viewer state
  const [spectrum2D, setSpectrum2D] = useState<Spectrum2DDataResponse | null>(null);
  const [isLoading2D, setIsLoading2D] = useState(false);

  // Debug: log molecule state changes in React
  useEffect(() => {
    console.log('[React] molecule state changed:', molecule);
    console.log('[React] molecule?.num_residues:', molecule?.num_residues);
  }, [molecule]);

  // Load realistic demo data using BMRB statistics
  const loadDemoData = useCallback(async () => {
    setIsLoadingDemo(true);
    try {
      // Generate realistic test data from BMRB statistics
      // 1H creates actual spectrum + ground truth, others create ground truth peak lists
      // All peaks are stored in backend state for assignment algorithm
      const result = await api.generateTestData(
        'ACDEFGHIKLMNPQRSTVWY', // All 20 amino acids
        'Demo Peptide',
        ['1H', 'HSQC', 'CHSQC', 'TOCSY', 'NOESY'], // Generate all experiments for full assignment
        {
          spectrometer_frequency_mhz: 600,
          num_points_1d: 8192,
          line_width_hz: 5.0,
        }
      );

      console.log('Generated test data:', result.summary);
      console.log('Spectra IDs:', result.spectra_ids);
      console.log('Ground truth peaks:');
      console.log('  1H:', result.ground_truth['1H']?.peaks.length ?? 0, 'peaks');
      console.log('  HSQC (15N):', result.ground_truth['HSQC']?.peaks.length ?? 0, 'peaks');
      console.log('  CHSQC (13C):', result.ground_truth['CHSQC']?.peaks.length ?? 0, 'peaks');
      console.log('  TOCSY:', result.ground_truth['TOCSY']?.peaks.length ?? 0, 'peaks');
      console.log('  NOESY:', result.ground_truth['NOESY']?.peaks.length ?? 0, 'peaks');

      // Store ground truth for validation
      setGroundTruth(result.ground_truth);

      // Refresh spectrum list and select the generated spectrum
      await fetchSpectraList();
      if (result.spectra_ids['1H']) {
        setActiveSpectrum(result.spectra_ids['1H']);
      }

      // Fetch the molecule info
      await fetchActiveMolecule();

      // Debug: log the molecule state after fetch
      console.log('After fetchActiveMolecule, molecule state:', useAssignmentStore.getState().molecule);
    } catch (err) {
      console.error('Failed to load demo data:', err);
    } finally {
      setIsLoadingDemo(false);
    }
  }, [fetchSpectraList, setActiveSpectrum, fetchActiveMolecule]);

  // Load a 2D spectrum demo
  const loadDemo2D = useCallback(async (experimentType: string) => {
    if (!molecule) {
      console.error('Load a molecule first (use Load Demo Data)');
      return;
    }
    setIsLoading2D(true);
    try {
      const result = await api.generateDemo2D(experimentType);
      console.log(`Generated ${experimentType} 2D spectrum:`, result.data.length, 'x', result.data[0]?.length, 'points');
      console.log('Noise floor:', result.noise_floor);
      console.log('Number of peaks:', result.num_peaks);
      setSpectrum2D(result);
    } catch (err) {
      console.error('Failed to generate 2D spectrum:', err);
    } finally {
      setIsLoading2D(false);
    }
  }, [molecule]);

  // Import spectrum file
  const [isImporting, setIsImporting] = useState(false);

  const handleImportFile = useCallback(async () => {
    setIsImporting(true);
    try {
      const result = await browseAndImportSpectrum();
      if (result) {
        console.log('Imported spectrum:', result);
        await fetchSpectraList();
        setActiveSpectrum(result.spectrum_id);
      }
    } catch (err) {
      console.error('Failed to import spectrum:', err);
    } finally {
      setIsImporting(false);
    }
  }, [fetchSpectraList, setActiveSpectrum]);

  const handleImportBruker = useCallback(async () => {
    setIsImporting(true);
    try {
      const result = await browseAndImportBruker();
      if (result) {
        console.log('Imported Bruker spectrum:', result);
        await fetchSpectraList();
        setActiveSpectrum(result.spectrum_id);
      }
    } catch (err) {
      console.error('Failed to import Bruker spectrum:', err);
    } finally {
      setIsImporting(false);
    }
  }, [fetchSpectraList, setActiveSpectrum]);

  // Run assignment handler - uses ground truth shift lists if available
  const handleRunAssignment = useCallback(async () => {
    // If we have ground truth, create shift lists from it first
    let activeListId: string | null = null;

    // Create shift list from 15N-HSQC (backbone N-H correlations)
    if (_groundTruth?.['HSQC']) {
      try {
        activeListId = await api.createShiftListFromGroundTruth(
          _groundTruth['HSQC'],
          'HSQC_15N_ground_truth'
        );
        console.log('Created 15N-HSQC shift list:', activeListId);
      } catch (err) {
        console.warn('Could not create 15N-HSQC shift list:', err);
      }
    }

    // Create shift list from 13C-HSQC (C-H correlations - CA, CB, sidechain)
    if (_groundTruth?.['CHSQC']) {
      try {
        const chsqcListId = await api.createShiftListFromGroundTruth(
          _groundTruth['CHSQC'],
          'CHSQC_13C_ground_truth'
        );
        console.log('Created 13C-HSQC shift list:', chsqcListId);
        // Use CHSQC as active - has carbon assignments
        activeListId = chsqcListId;
      } catch (err) {
        console.warn('Could not create 13C-HSQC shift list:', err);
      }
    }

    // Also add 1H proton shifts (H, HA, etc.)
    if (_groundTruth?.['1H']) {
      try {
        const protonListId = await api.createShiftListFromGroundTruth(
          _groundTruth['1H'],
          '1H_ground_truth'
        );
        console.log('Created 1H shift list:', protonListId);
        // Use the proton list as active (has more complete backbone coverage)
        activeListId = protonListId;
      } catch (err) {
        console.warn('Could not create 1H shift list:', err);
      }
    }

    // Set the active shift list so assignment uses it
    if (activeListId) {
      setActiveShiftList(activeListId);
    }

    await runGlobalAssignment();
    // Switch to assignments tab when done
    setRightPanelTab('assignments');
  }, [runGlobalAssignment, setRightPanelTab, _groundTruth, setActiveShiftList]);

  return (
    <div className="h-screen flex flex-col bg-slate-900">
      {/* Top toolbar */}
      <header className="flex items-center justify-between px-4 py-2 bg-slate-800 border-b border-slate-700">
        <div className="flex items-center gap-4">
          <h1 className="text-lg font-bold text-slate-100">NMRaster</h1>
          <span className="text-xs text-slate-500">Next-gen NMR Analysis</span>
        </div>

        <div className="flex items-center gap-2">
          {/* Import buttons */}
          <div className="flex items-center gap-1 mr-2">
            <button
              onClick={handleImportFile}
              disabled={isImporting}
              className="btn btn-secondary text-sm disabled:opacity-50"
              title="Import NMRPipe spectrum file"
            >
              {isImporting ? 'Importing...' : 'Import File'}
            </button>
            <button
              onClick={handleImportBruker}
              disabled={isImporting}
              className="btn btn-secondary text-sm disabled:opacity-50"
              title="Import Bruker experiment directory"
            >
              Bruker
            </button>
          </div>

          <div className="w-px h-6 bg-slate-600" />

          <button
            onClick={loadDemoData}
            disabled={isLoadingDemo}
            className="btn btn-primary text-sm disabled:opacity-50"
          >
            {isLoadingDemo ? 'Loading...' : 'Load Demo Data'}
          </button>

          <button
            onClick={handleRunAssignment}
            disabled={isAssigning || !molecule}
            className="btn btn-secondary text-sm disabled:opacity-50"
            title={!molecule ? 'Load a molecule first' : 'Run global assignment'}
          >
            {isAssigning ? 'Assigning...' : 'Run Assignment'}
          </button>

          {/* 2D spectrum demos */}
          <div className="flex items-center gap-1 ml-2">
            <span className="text-xs text-slate-400">2D:</span>
            <button
              onClick={() => loadDemo2D('HSQC')}
              disabled={isLoading2D || !molecule}
              className="btn btn-secondary text-xs px-2 py-1 disabled:opacity-50"
              title="Generate 15N-HSQC"
            >
              HSQC
            </button>
            <button
              onClick={() => loadDemo2D('CHSQC')}
              disabled={isLoading2D || !molecule}
              className="btn btn-secondary text-xs px-2 py-1 disabled:opacity-50"
              title="Generate 13C-HSQC"
            >
              13C
            </button>
            <button
              onClick={() => loadDemo2D('TOCSY')}
              disabled={isLoading2D || !molecule}
              className="btn btn-secondary text-xs px-2 py-1 disabled:opacity-50"
              title="Generate TOCSY"
            >
              TOCSY
            </button>
            <button
              onClick={() => loadDemo2D('NOESY')}
              disabled={isLoading2D || !molecule}
              className="btn btn-secondary text-xs px-2 py-1 disabled:opacity-50"
              title="Generate NOESY"
            >
              NOESY
            </button>
          </div>

          {/* Panel toggles */}
          <div className="flex items-center gap-1 ml-4">
            <button
              onClick={toggleSpectrumList}
              className={`btn text-xs px-2 py-1 ${showSpectrumList ? 'btn-primary' : 'btn-secondary'}`}
              title="Toggle Spectrum List"
            >
              Spectra
            </button>
            <button
              onClick={togglePeakList}
              className={`btn text-xs px-2 py-1 ${showPeakList ? 'btn-primary' : 'btn-secondary'}`}
              title="Toggle Right Panel"
            >
              Panel
            </button>
            <button
              onClick={toggleSequence}
              className={`btn text-xs px-2 py-1 ${showSequence ? 'btn-primary' : 'btn-secondary'}`}
              title="Toggle Sequence"
            >
              Sequence
            </button>
            <button
              onClick={toggleSpinSystemView}
              className={`btn text-xs px-2 py-1 ${showSpinSystemView ? 'btn-primary' : 'btn-secondary'}`}
              title="Toggle Spin System"
            >
              SpinSys
            </button>
          </div>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left sidebar - Spectrum list */}
        {showSpectrumList && (
          <aside className="w-64 border-r border-slate-700 overflow-hidden">
            <SpectrumList />
          </aside>
        )}

        {/* Center - Spectrum viewer */}
        <main className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 p-2 overflow-hidden">
            <SpectrumViewer />
          </div>

          {/* Bottom - Sequence viewer */}
          {showSequence && (
            <div className="border-t border-slate-700">
              <SequenceViewer />
            </div>
          )}
        </main>

        {/* Right sidebar - Tabbed panel */}
        {showPeakList && (
          <aside className="w-80 border-l border-slate-700 flex flex-col overflow-hidden">
            {/* Tab header */}
            <div className="flex border-b border-slate-700">
              <button
                className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
                  rightPanelTab === 'peaks'
                    ? 'bg-slate-700 text-slate-100'
                    : 'bg-slate-800 text-slate-400 hover:text-slate-200'
                }`}
                onClick={() => setRightPanelTab('peaks')}
              >
                Peaks
              </button>
              <button
                className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
                  rightPanelTab === 'assignments'
                    ? 'bg-slate-700 text-slate-100'
                    : 'bg-slate-800 text-slate-400 hover:text-slate-200'
                }`}
                onClick={() => setRightPanelTab('assignments')}
              >
                Assignments
                {assignmentResults.length > 0 && (
                  <span className="ml-1 text-xs text-slate-400">
                    ({assignmentResults.length})
                  </span>
                )}
              </button>
            </div>

            {/* Tab content */}
            <div className="flex-1 overflow-hidden">
              {rightPanelTab === 'peaks' ? <PeakList /> : <AssignmentTable />}
            </div>

            {/* Spin System View (below tabs) */}
            {showSpinSystemView && (
              <div className="h-64 border-t border-slate-700">
                <SpinSystemView />
              </div>
            )}
          </aside>
        )}
      </div>

      {/* Status bar */}
      <footer className="px-4 py-1 bg-slate-800 border-t border-slate-700 text-xs text-slate-500">
        <div className="flex items-center justify-between">
          <span>
            {molecule ? `Molecule: ${molecule.name} (${molecule.num_residues} residues)` : 'Ready'}
          </span>
          <span>
            {assignmentResults.length > 0
              ? `${assignmentResults.length} assignments`
              : 'NMRaster v0.1.0'}
          </span>
        </div>
      </footer>

      {/* 2D Spectrum Viewer Modal */}
      {spectrum2D && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="w-[90vw] h-[85vh] max-w-6xl">
            <SpectrumViewer2D
              spectrum={spectrum2D}
              title={`${spectrum2D.experiment_type} - ${molecule?.name || 'Demo'}`}
              onClose={() => setSpectrum2D(null)}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
