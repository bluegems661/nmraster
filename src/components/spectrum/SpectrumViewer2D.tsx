/**
 * 2D Spectrum viewer with contour controls and interaction.
 * Wraps SpectrumCanvas2D with UI controls for contour levels, zoom, etc.
 */

import { useState, useCallback, useMemo, useEffect } from 'react';
import { SpectrumCanvas2D, type ViewBounds2D } from './SpectrumCanvas2D';
import type { Spectrum2DDataResponse } from '../../types/tauri';

interface SpectrumViewer2DProps {
  spectrum: Spectrum2DDataResponse | null;
  title?: string;
  onClose?: () => void;
}

export function SpectrumViewer2D({ spectrum, title, onClose }: SpectrumViewer2DProps) {
  // Contour settings
  const [contourLevels, setContourLevels] = useState(8);
  const [contourBase, setContourBase] = useState(5);
  const [showPositive, setShowPositive] = useState(true);
  const [showNegative, setShowNegative] = useState(true);

  // View state
  const [viewBounds, setViewBounds] = useState<ViewBounds2D | null>(null);

  // Mouse position
  const [mouseCoords, setMouseCoords] = useState<{ f2: number; f1: number; intensity: number } | null>(null);

  // Initialize view bounds when spectrum loads
  useEffect(() => {
    if (spectrum) {
      const f1 = spectrum.ppm_axis_f1;
      const f2 = spectrum.ppm_axis_f2;

      setViewBounds({
        xMin: Math.min(...f2),
        xMax: Math.max(...f2),
        yMin: Math.min(...f1),
        yMax: Math.max(...f1),
      });
    }
  }, [spectrum]);

  // Full view bounds for reset
  const fullBounds = useMemo<ViewBounds2D | null>(() => {
    if (!spectrum) return null;

    const f1 = spectrum.ppm_axis_f1;
    const f2 = spectrum.ppm_axis_f2;

    return {
      xMin: Math.min(...f2),
      xMax: Math.max(...f2),
      yMin: Math.min(...f1),
      yMax: Math.max(...f1),
    };
  }, [spectrum]);

  const handleViewChange = useCallback((bounds: ViewBounds2D) => {
    setViewBounds(bounds);
  }, []);

  const handleMouseMove = useCallback((f2Ppm: number, f1Ppm: number, intensity: number) => {
    setMouseCoords({ f2: f2Ppm, f1: f1Ppm, intensity });
  }, []);

  const handleResetView = useCallback(() => {
    if (fullBounds) {
      setViewBounds(fullBounds);
    }
  }, [fullBounds]);

  const formatIntensity = (value: number): string => {
    const abs = Math.abs(value);
    if (abs >= 1e6) return (value / 1e6).toFixed(2) + 'M';
    if (abs >= 1e3) return (value / 1e3).toFixed(2) + 'K';
    if (abs >= 1) return value.toFixed(2);
    if (abs >= 0.001) return value.toFixed(4);
    return value.toExponential(2);
  };

  return (
    <div className="flex flex-col h-full bg-slate-900 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-slate-800 border-b border-slate-700">
        <div className="flex items-center gap-4">
          <h3 className="text-sm font-medium text-slate-200">
            {title || spectrum?.experiment_type || '2D Spectrum'}
          </h3>
          {spectrum && (
            <span className="text-xs text-slate-400">
              {spectrum.num_peaks} peaks
            </span>
          )}
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-200 transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4 px-4 py-2 bg-slate-850 border-b border-slate-700 text-xs">
        {/* Contour levels */}
        <div className="flex items-center gap-2">
          <label className="text-slate-400">Levels:</label>
          <input
            type="range"
            min={4}
            max={12}
            value={contourLevels}
            onChange={(e) => setContourLevels(parseInt(e.target.value))}
            className="w-20 h-1 bg-slate-600 rounded-lg appearance-none cursor-pointer"
          />
          <span className="text-slate-300 w-4">{contourLevels}</span>
        </div>

        {/* Base threshold */}
        <div className="flex items-center gap-2">
          <label className="text-slate-400">Threshold:</label>
          <input
            type="range"
            min={1}
            max={20}
            value={contourBase}
            onChange={(e) => setContourBase(parseInt(e.target.value))}
            className="w-20 h-1 bg-slate-600 rounded-lg appearance-none cursor-pointer"
          />
          <span className="text-slate-300 w-6">{contourBase}x</span>
        </div>

        {/* Show positive/negative */}
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-1 cursor-pointer">
            <input
              type="checkbox"
              checked={showPositive}
              onChange={(e) => setShowPositive(e.target.checked)}
              className="rounded border-slate-500 bg-slate-700 text-blue-500 focus:ring-blue-500"
            />
            <span className="text-blue-400">+ve</span>
          </label>
          <label className="flex items-center gap-1 cursor-pointer">
            <input
              type="checkbox"
              checked={showNegative}
              onChange={(e) => setShowNegative(e.target.checked)}
              className="rounded border-slate-500 bg-slate-700 text-red-500 focus:ring-red-500"
            />
            <span className="text-red-400">-ve</span>
          </label>
        </div>

        {/* Reset view */}
        <button
          onClick={handleResetView}
          className="px-2 py-1 text-slate-300 bg-slate-700 hover:bg-slate-600 rounded transition-colors"
        >
          Reset View
        </button>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Mouse coordinates */}
        {mouseCoords && (
          <div className="text-slate-400 font-mono">
            F2: <span className="text-slate-200">{mouseCoords.f2.toFixed(3)}</span> ppm |
            F1: <span className="text-slate-200">{mouseCoords.f1.toFixed(2)}</span> ppm |
            I: <span className="text-slate-200">{formatIntensity(mouseCoords.intensity)}</span>
          </div>
        )}
      </div>

      {/* Canvas */}
      <div className="flex-1 min-h-0">
        <SpectrumCanvas2D
          spectrum={spectrum}
          viewBounds={viewBounds}
          contourLevels={contourLevels}
          contourBase={contourBase}
          showPositive={showPositive}
          showNegative={showNegative}
          onViewChange={handleViewChange}
          onMouseMove={handleMouseMove}
        />
      </div>

      {/* Footer with info */}
      {spectrum && (() => {
        // Calculate max safely (avoid stack overflow with spread operator)
        let maxVal = 0;
        for (const row of spectrum.data) {
          for (const v of row) {
            if (v > maxVal) maxVal = v;
          }
        }
        return (
          <div className="px-4 py-1 bg-slate-800 border-t border-slate-700 text-xs text-slate-500">
            {spectrum.data.length} x {spectrum.data[0]?.length || 0} points |
            Noise floor: {spectrum.noise_floor > 0 ? spectrum.noise_floor.toExponential(2) : 'auto'} |
            Max: {maxVal.toFixed(4)} |
            Drag to pan, scroll to zoom
          </div>
        );
      })()}
    </div>
  );
}
