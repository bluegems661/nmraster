/**
 * Sidebar list of loaded spectra.
 */

import { useEffect } from 'react';
import { useSpectrumStore } from '../../stores/spectrumStore';

export function SpectrumList() {
  const {
    spectraList,
    activeSpectrumId,
    setActiveSpectrum,
    fetchSpectraList,
    isLoading,
  } = useSpectrumStore();

  // Fetch spectra list on mount
  useEffect(() => {
    fetchSpectraList();
  }, [fetchSpectraList]);

  return (
    <div className="panel h-full flex flex-col">
      {/* Header */}
      <div className="p-2 border-b border-slate-700">
        <h3 className="text-sm font-medium text-slate-200">Spectra</h3>
      </div>

      {/* List */}
      <div className="flex-1 overflow-auto">
        {isLoading ? (
          <div className="p-4 text-center text-slate-500 text-sm">
            Loading...
          </div>
        ) : spectraList.length === 0 ? (
          <div className="p-4 text-center text-slate-500 text-sm">
            No spectra loaded.
            <br />
            <span className="text-xs">
              Use the demo button to load sample data.
            </span>
          </div>
        ) : (
          <ul className="p-1">
            {spectraList.map((spectrum) => (
              <li key={spectrum.id}>
                <button
                  onClick={() => setActiveSpectrum(spectrum.id)}
                  className={`
                    w-full text-left p-2 rounded text-sm transition-colors
                    ${
                      spectrum.id === activeSpectrumId
                        ? 'bg-blue-900/50 text-blue-200'
                        : 'hover:bg-slate-700/50 text-slate-300'
                    }
                  `}
                >
                  <div className="font-medium truncate">{spectrum.name}</div>
                  <div className="text-xs text-slate-500 flex items-center gap-2">
                    <span>{spectrum.experiment_type}</span>
                    <span>•</span>
                    <span>{spectrum.dimensions}D</span>
                    <span>•</span>
                    <span>{spectrum.num_points.toLocaleString()} pts</span>
                    {spectrum.is_processed && (
                      <>
                        <span>•</span>
                        <span className="text-green-500">FFT</span>
                      </>
                    )}
                  </div>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
