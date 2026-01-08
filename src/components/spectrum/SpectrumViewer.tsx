/**
 * Spectrum viewer container with toolbar and info display.
 */

import { useState, useCallback } from 'react';
import { SpectrumCanvas } from './SpectrumCanvas';
import { useSpectrumStore } from '../../stores/spectrumStore';
import { useUIStore } from '../../stores/uiStore';

export function SpectrumViewer() {
  const {
    spectra,
    peaks,
    activeSpectrumId,
    selectedPeakId,
    viewBounds,
    toolMode,
    setViewBounds,
    resetView,
    zoomIn,
    zoomOut,
    selectPeak,
    setToolMode,
    pickPeaks,
    clearPeaks,
  } = useSpectrumStore();

  const { spectrumLineWidth } = useUIStore();

  // Mouse position display
  const [mousePos, setMousePos] = useState<{ ppm: number; intensity: number } | null>(null);

  // Peak picking dialog state
  const [showPeakPickDialog, setShowPeakPickDialog] = useState(false);
  const [peakPickSnr, setPeakPickSnr] = useState('5');

  const activeSpectrum = activeSpectrumId ? spectra.get(activeSpectrumId) ?? null : null;
  const activePeaks = activeSpectrumId ? peaks.get(activeSpectrumId) ?? [] : [];

  const handleMouseMove = useCallback((ppm: number, intensity: number) => {
    setMousePos({ ppm, intensity });
  }, []);

  const handlePeakPick = useCallback(async () => {
    if (!activeSpectrumId) return;
    const snr = parseFloat(peakPickSnr);
    if (isNaN(snr) || snr <= 0) return;

    await pickPeaks(activeSpectrumId, snr);
    setShowPeakPickDialog(false);
  }, [activeSpectrumId, peakPickSnr, pickPeaks]);

  const handleClearPeaks = useCallback(async () => {
    if (!activeSpectrumId) return;
    await clearPeaks(activeSpectrumId);
  }, [activeSpectrumId, clearPeaks]);

  return (
    <div className="flex flex-col h-full panel">
      {/* Toolbar */}
      <div className="flex items-center gap-2 p-2 border-b border-slate-700">
        {/* Spectrum info */}
        <div className="flex-1 text-sm text-slate-300 truncate">
          {activeSpectrum ? (
            <span>
              <span className="font-medium text-slate-100">{activeSpectrum.name}</span>
              <span className="mx-2 text-slate-500">|</span>
              <span>{activeSpectrum.experiment_type}</span>
              <span className="mx-2 text-slate-500">|</span>
              <span>{activeSpectrum.real.length.toLocaleString()} pts</span>
            </span>
          ) : (
            <span className="text-slate-500">No spectrum selected</span>
          )}
        </div>

        {/* Zoom controls */}
        <div className="flex items-center gap-1">
          <button
            onClick={zoomIn}
            disabled={!activeSpectrum}
            className="btn btn-secondary text-sm px-2 py-1 disabled:opacity-50"
            title="Zoom In"
          >
            +
          </button>
          <button
            onClick={zoomOut}
            disabled={!activeSpectrum}
            className="btn btn-secondary text-sm px-2 py-1 disabled:opacity-50"
            title="Zoom Out"
          >
            −
          </button>
          <button
            onClick={resetView}
            disabled={!activeSpectrum}
            className="btn btn-secondary text-sm px-2 py-1 disabled:opacity-50"
            title="Reset View"
          >
            ⊡
          </button>
        </div>

        {/* Separator */}
        <div className="w-px h-6 bg-slate-600" />

        {/* Peak picking */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => setShowPeakPickDialog(true)}
            disabled={!activeSpectrum}
            className="btn btn-secondary text-sm disabled:opacity-50"
            title="Pick Peaks"
          >
            Pick Peaks
          </button>
          <button
            onClick={handleClearPeaks}
            disabled={!activeSpectrumId || activePeaks.length === 0}
            className="btn btn-secondary text-sm disabled:opacity-50"
            title="Clear Peaks"
          >
            Clear
          </button>
        </div>

        {/* Tool mode selector */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => setToolMode('select')}
            className={`btn text-sm px-2 py-1 ${
              toolMode === 'select' ? 'btn-primary' : 'btn-secondary'
            }`}
            title="Select Mode"
          >
            ↖
          </button>
          <button
            onClick={() => setToolMode('zoom')}
            className={`btn text-sm px-2 py-1 ${
              toolMode === 'zoom' ? 'btn-primary' : 'btn-secondary'
            }`}
            title="Zoom Mode"
          >
            ⌕
          </button>
        </div>
      </div>

      {/* Canvas */}
      <div className="flex-1 relative">
        <SpectrumCanvas
          spectrum={activeSpectrum}
          peaks={activePeaks}
          viewBounds={viewBounds}
          selectedPeakId={selectedPeakId}
          lineWidth={spectrumLineWidth}
          onPeakSelect={selectPeak}
          onViewChange={setViewBounds}
          onMouseMove={handleMouseMove}
        />

        {/* Mouse position overlay */}
        {mousePos && activeSpectrum && (
          <div className="absolute bottom-2 right-2 bg-slate-800/90 px-2 py-1 rounded text-xs text-slate-300 font-mono">
            {mousePos.ppm.toFixed(3)} ppm, {formatIntensity(mousePos.intensity)}
          </div>
        )}
      </div>

      {/* Peak Pick Dialog */}
      {showPeakPickDialog && (
        <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-slate-800 rounded-lg p-4 shadow-xl border border-slate-600 w-72">
            <h3 className="text-lg font-medium text-slate-100 mb-4">Pick Peaks</h3>

            <div className="mb-4">
              <label className="block text-sm text-slate-300 mb-1">
                Minimum SNR
              </label>
              <input
                type="number"
                value={peakPickSnr}
                onChange={(e) => setPeakPickSnr(e.target.value)}
                className="input w-full"
                min="1"
                step="1"
              />
            </div>

            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowPeakPickDialog(false)}
                className="btn btn-secondary"
              >
                Cancel
              </button>
              <button onClick={handlePeakPick} className="btn btn-primary">
                Pick
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function formatIntensity(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1e6) return (value / 1e6).toFixed(2) + 'M';
  if (abs >= 1e3) return (value / 1e3).toFixed(2) + 'K';
  if (abs >= 1) return value.toFixed(1);
  return value.toExponential(2);
}
