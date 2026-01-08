/**
 * Zustand store for spectrum data and visualization state.
 */

import { create } from 'zustand';
import type {
  SpectrumDataResponse,
  SpectrumInfo,
  PickedPeak,
  ViewBounds,
  ToolMode,
} from '../types/tauri';
import * as api from '../lib/tauri';

interface SpectrumState {
  // Data
  spectra: Map<string, SpectrumDataResponse>;
  spectraList: SpectrumInfo[];
  peaks: Map<string, PickedPeak[]>;

  // Selection
  activeSpectrumId: string | null;
  selectedPeakId: string | null;

  // View state
  viewBounds: ViewBounds | null;
  autoScale: boolean;

  // Tool state
  toolMode: ToolMode;

  // Loading state
  isLoading: boolean;
  error: string | null;

  // Actions - Data
  fetchSpectraList: () => Promise<void>;
  fetchSpectrum: (id: string) => Promise<void>;
  fetchPeaks: (spectrumId: string) => Promise<void>;

  // Actions - Selection
  setActiveSpectrum: (id: string | null) => void;
  selectPeak: (id: string | null) => void;

  // Actions - View
  setViewBounds: (bounds: ViewBounds) => void;
  resetView: () => void;
  setAutoScale: (enabled: boolean) => void;
  zoomIn: () => void;
  zoomOut: () => void;

  // Actions - Tools
  setToolMode: (mode: ToolMode) => void;

  // Actions - Analysis
  pickPeaks: (spectrumId: string, minSnr: number) => Promise<void>;
  clearPeaks: (spectrumId: string) => Promise<void>;
}

export const useSpectrumStore = create<SpectrumState>((set, get) => ({
  // Initial state
  spectra: new Map(),
  spectraList: [],
  peaks: new Map(),
  activeSpectrumId: null,
  selectedPeakId: null,
  viewBounds: null,
  autoScale: true,
  toolMode: 'select',
  isLoading: false,
  error: null,

  // Fetch list of all spectra
  fetchSpectraList: async () => {
    set({ isLoading: true, error: null });
    try {
      const list = await api.listSpectra();
      set({ spectraList: list, isLoading: false });
    } catch (err) {
      set({ error: String(err), isLoading: false });
    }
  },

  // Fetch full spectrum data
  fetchSpectrum: async (id: string) => {
    set({ isLoading: true, error: null });
    try {
      const spectrum = await api.getSpectrum1D(id);
      if (spectrum) {
        const spectra = new Map(get().spectra);
        spectra.set(id, spectrum);
        set({ spectra, isLoading: false });

        // Auto-set view bounds if autoScale is on
        if (get().autoScale) {
          get().resetView();
        }
      }
    } catch (err) {
      set({ error: String(err), isLoading: false });
    }
  },

  // Fetch peaks for a spectrum
  fetchPeaks: async (spectrumId: string) => {
    try {
      const spectrumPeaks = await api.getSpectrumPeaks(spectrumId);
      // Convert PeakInfo[] to PickedPeak[] format
      const pickedPeaks: PickedPeak[] = spectrumPeaks.map((p) => ({
        id: p.id,
        ppm: p.position_ppm[0] ?? 0,
        intensity: p.intensity,
        snr: 0, // Not available from PeakInfo
      }));
      const peaks = new Map(get().peaks);
      peaks.set(spectrumId, pickedPeaks);
      set({ peaks });
    } catch (err) {
      set({ error: String(err) });
    }
  },

  // Set active spectrum
  setActiveSpectrum: (id: string | null) => {
    set({ activeSpectrumId: id, selectedPeakId: null });
    if (id && !get().spectra.has(id)) {
      get().fetchSpectrum(id);
    }
    if (id) {
      get().fetchPeaks(id);
    }
    if (get().autoScale) {
      get().resetView();
    }
  },

  // Select a peak
  selectPeak: (id: string | null) => {
    set({ selectedPeakId: id });
  },

  // Set view bounds
  setViewBounds: (bounds: ViewBounds) => {
    set({ viewBounds: bounds, autoScale: false });
  },

  // Reset view to full spectrum
  resetView: () => {
    const { activeSpectrumId, spectra } = get();
    if (!activeSpectrumId) {
      set({ viewBounds: null });
      return;
    }

    const spectrum = spectra.get(activeSpectrumId);
    if (!spectrum) {
      set({ viewBounds: null });
      return;
    }

    const ppmAxis = spectrum.ppm_axis;
    const real = spectrum.real;

    const xMin = Math.min(...ppmAxis);
    const xMax = Math.max(...ppmAxis);
    const yMin = Math.min(...real);
    const yMax = Math.max(...real);

    // Add 5% padding
    const xPad = (xMax - xMin) * 0.05;
    const yPad = (yMax - yMin) * 0.1;

    set({
      viewBounds: {
        xMin: xMin - xPad,
        xMax: xMax + xPad,
        yMin: yMin - yPad,
        yMax: yMax + yPad,
      },
      autoScale: true,
    });
  },

  // Toggle auto-scale
  setAutoScale: (enabled: boolean) => {
    set({ autoScale: enabled });
    if (enabled) {
      get().resetView();
    }
  },

  // Zoom in (center)
  zoomIn: () => {
    const { viewBounds } = get();
    if (!viewBounds) return;

    const xCenter = (viewBounds.xMin + viewBounds.xMax) / 2;
    const yCenter = (viewBounds.yMin + viewBounds.yMax) / 2;
    const xRange = (viewBounds.xMax - viewBounds.xMin) * 0.4; // 40% of current
    const yRange = (viewBounds.yMax - viewBounds.yMin) * 0.4;

    set({
      viewBounds: {
        xMin: xCenter - xRange,
        xMax: xCenter + xRange,
        yMin: yCenter - yRange,
        yMax: yCenter + yRange,
      },
      autoScale: false,
    });
  },

  // Zoom out
  zoomOut: () => {
    const { viewBounds } = get();
    if (!viewBounds) return;

    const xCenter = (viewBounds.xMin + viewBounds.xMax) / 2;
    const yCenter = (viewBounds.yMin + viewBounds.yMax) / 2;
    const xRange = (viewBounds.xMax - viewBounds.xMin) * 0.8; // 80% of current
    const yRange = (viewBounds.yMax - viewBounds.yMin) * 0.8;

    set({
      viewBounds: {
        xMin: xCenter - xRange,
        xMax: xCenter + xRange,
        yMin: yCenter - yRange,
        yMax: yCenter + yRange,
      },
      autoScale: false,
    });
  },

  // Set tool mode
  setToolMode: (mode: ToolMode) => {
    set({ toolMode: mode });
  },

  // Pick peaks
  pickPeaks: async (spectrumId: string, minSnr: number) => {
    set({ isLoading: true, error: null });
    try {
      console.log('Picking peaks for spectrum:', spectrumId, 'with SNR:', minSnr);
      const pickedPeaks = await api.pickPeaks1D(spectrumId, { min_snr: minSnr });
      console.log('Picked peaks:', pickedPeaks);
      const peaks = new Map(get().peaks);
      peaks.set(spectrumId, pickedPeaks);
      set({ peaks, isLoading: false });
    } catch (err) {
      console.error('Peak picking failed:', err);
      set({ error: String(err), isLoading: false });
    }
  },

  // Clear peaks
  clearPeaks: async (spectrumId: string) => {
    try {
      await api.clearSpectrumPeaks(spectrumId);
      const peaks = new Map(get().peaks);
      peaks.delete(spectrumId);
      set({ peaks, selectedPeakId: null });
    } catch (err) {
      set({ error: String(err) });
    }
  },
}));
