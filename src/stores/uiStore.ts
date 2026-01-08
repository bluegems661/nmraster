/**
 * Zustand store for UI state (panels, preferences).
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface UIState {
  // Panel visibility
  showSpectrumList: boolean;
  showPeakList: boolean;
  showSequence: boolean;
  showAssignmentTable: boolean;
  showSpinSystemView: boolean;

  // Right panel tab
  rightPanelTab: 'peaks' | 'assignments';

  // Panel actions
  toggleSpectrumList: () => void;
  togglePeakList: () => void;
  toggleSequence: () => void;
  toggleAssignmentTable: () => void;
  toggleSpinSystemView: () => void;
  setRightPanelTab: (tab: 'peaks' | 'assignments') => void;

  // Preferences
  darkMode: boolean;
  spectrumLineWidth: number;
  peakLabelSize: number;

  // Preference actions
  setDarkMode: (enabled: boolean) => void;
  setSpectrumLineWidth: (width: number) => void;
  setPeakLabelSize: (size: number) => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      // Initial panel visibility
      showSpectrumList: true,
      showPeakList: true,
      showSequence: true,
      showAssignmentTable: false,
      showSpinSystemView: true,

      // Right panel tab
      rightPanelTab: 'peaks',

      // Panel toggles
      toggleSpectrumList: () => set((s) => ({ showSpectrumList: !s.showSpectrumList })),
      togglePeakList: () => set((s) => ({ showPeakList: !s.showPeakList })),
      toggleSequence: () => set((s) => ({ showSequence: !s.showSequence })),
      toggleAssignmentTable: () => set((s) => ({ showAssignmentTable: !s.showAssignmentTable })),
      toggleSpinSystemView: () => set((s) => ({ showSpinSystemView: !s.showSpinSystemView })),
      setRightPanelTab: (tab) => set({ rightPanelTab: tab }),

      // Default preferences
      darkMode: true,
      spectrumLineWidth: 1,
      peakLabelSize: 10,

      // Preference setters
      setDarkMode: (enabled) => set({ darkMode: enabled }),
      setSpectrumLineWidth: (width) => set({ spectrumLineWidth: width }),
      setPeakLabelSize: (size) => set({ peakLabelSize: size }),
    }),
    {
      name: 'nmraster-ui', // localStorage key
      partialize: (state) => ({
        // Only persist preferences, not panel visibility
        darkMode: state.darkMode,
        spectrumLineWidth: state.spectrumLineWidth,
        peakLabelSize: state.peakLabelSize,
      }),
    }
  )
);
