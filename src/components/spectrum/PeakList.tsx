/**
 * Sortable table of picked peaks.
 */

import { useState, useMemo } from 'react';
import { useSpectrumStore } from '../../stores/spectrumStore';
import type { PickedPeak } from '../../types/tauri';

type SortField = 'ppm' | 'intensity' | 'snr';
type SortDirection = 'asc' | 'desc';

export function PeakList() {
  const { peaks, activeSpectrumId, selectedPeakId, selectPeak } = useSpectrumStore();

  const [sortField, setSortField] = useState<SortField>('ppm');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [filter, setFilter] = useState('');

  const activePeaks = activeSpectrumId ? peaks.get(activeSpectrumId) ?? [] : [];

  // Filter and sort peaks
  const sortedPeaks = useMemo(() => {
    let filtered = activePeaks;

    // Apply filter (search by ppm)
    if (filter) {
      const filterNum = parseFloat(filter);
      if (!isNaN(filterNum)) {
        filtered = filtered.filter((p) =>
          Math.abs(p.ppm - filterNum) < 0.5 ||
          p.ppm.toFixed(2).includes(filter)
        );
      }
    }

    // Sort
    const sorted = [...filtered].sort((a, b) => {
      let comparison = 0;
      switch (sortField) {
        case 'ppm':
          comparison = a.ppm - b.ppm;
          break;
        case 'intensity':
          comparison = a.intensity - b.intensity;
          break;
        case 'snr':
          comparison = a.snr - b.snr;
          break;
      }
      return sortDirection === 'asc' ? comparison : -comparison;
    });

    return sorted;
  }, [activePeaks, sortField, sortDirection, filter]);

  const handleSort = (field: SortField) => {
    if (field === sortField) {
      setSortDirection((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const SortIcon = ({ field }: { field: SortField }) => {
    if (field !== sortField) return <span className="text-slate-600">⇅</span>;
    return sortDirection === 'asc' ? (
      <span className="text-blue-400">↑</span>
    ) : (
      <span className="text-blue-400">↓</span>
    );
  };

  if (!activeSpectrumId) {
    return (
      <div className="panel h-full flex items-center justify-center text-slate-500 text-sm">
        No spectrum selected
      </div>
    );
  }

  return (
    <div className="panel h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-2 border-b border-slate-700">
        <h3 className="text-sm font-medium text-slate-200">
          Peaks ({activePeaks.length})
        </h3>
        <input
          type="text"
          placeholder="Filter by ppm..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="input text-xs w-32"
        />
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-slate-800">
            <tr className="text-left text-slate-400 text-xs">
              <th className="p-2 w-8">#</th>
              <th
                className="p-2 cursor-pointer hover:text-slate-200"
                onClick={() => handleSort('ppm')}
              >
                PPM <SortIcon field="ppm" />
              </th>
              <th
                className="p-2 cursor-pointer hover:text-slate-200"
                onClick={() => handleSort('intensity')}
              >
                Intensity <SortIcon field="intensity" />
              </th>
              <th
                className="p-2 cursor-pointer hover:text-slate-200"
                onClick={() => handleSort('snr')}
              >
                SNR <SortIcon field="snr" />
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedPeaks.length === 0 ? (
              <tr>
                <td colSpan={4} className="p-4 text-center text-slate-500">
                  {activePeaks.length === 0
                    ? 'No peaks picked. Use "Pick Peaks" to detect peaks.'
                    : 'No peaks match filter.'}
                </td>
              </tr>
            ) : (
              sortedPeaks.map((peak, index) => (
                <PeakRow
                  key={peak.id}
                  peak={peak}
                  index={index + 1}
                  isSelected={peak.id === selectedPeakId}
                  onSelect={() => selectPeak(peak.id)}
                />
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

interface PeakRowProps {
  peak: PickedPeak;
  index: number;
  isSelected: boolean;
  onSelect: () => void;
}

function PeakRow({ peak, index, isSelected, onSelect }: PeakRowProps) {
  return (
    <tr
      className={`cursor-pointer hover:bg-slate-700/50 ${
        isSelected ? 'bg-blue-900/30' : ''
      }`}
      onClick={onSelect}
    >
      <td className="p-2 text-slate-500">{index}</td>
      <td className="p-2 font-mono text-slate-200">{peak.ppm.toFixed(3)}</td>
      <td className="p-2 font-mono text-slate-300">
        {formatIntensity(peak.intensity)}
      </td>
      <td className="p-2 font-mono text-slate-300">{peak.snr.toFixed(1)}</td>
    </tr>
  );
}

function formatIntensity(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1e6) return (value / 1e6).toFixed(2) + 'M';
  if (abs >= 1e3) return (value / 1e3).toFixed(2) + 'K';
  if (abs >= 1) return value.toFixed(1);
  return value.toExponential(2);
}
