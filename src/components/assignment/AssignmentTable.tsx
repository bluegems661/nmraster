/**
 * Sortable table of assignment results with confidence display.
 */

import { useState, useMemo } from 'react';
import { useAssignmentStore } from '../../stores/assignmentStore';
import type { AssignmentResult, ConfidenceLevel } from '../../types/tauri';

type SortField = 'residue' | 'atom' | 'shift' | 'confidence';
type SortDirection = 'asc' | 'desc';

const CONFIDENCE_COLORS: Record<ConfidenceLevel, string> = {
  high: 'text-green-400 bg-green-900/30',
  medium: 'text-amber-400 bg-amber-900/30',
  low: 'text-red-400 bg-red-900/30',
};

export function AssignmentTable() {
  const {
    assignmentResults,
    selectedResidueSeqCode,
    selectedAtomId,
    selectResidue,
    selectAtom,
    getConfidenceLevel,
  } = useAssignmentStore();

  const [sortField, setSortField] = useState<SortField>('residue');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');
  const [filter, setFilter] = useState('');
  const [showAlternatives, setShowAlternatives] = useState<string | null>(null);

  // Filter and sort results
  const sortedResults = useMemo(() => {
    let filtered = assignmentResults;

    // Apply filter (search by residue number or atom name)
    if (filter) {
      const filterLower = filter.toLowerCase();
      const filterNum = parseInt(filter, 10);
      filtered = filtered.filter(
        (r) =>
          r.atom_name.toLowerCase().includes(filterLower) ||
          r.residue_name.toLowerCase().includes(filterLower) ||
          (!isNaN(filterNum) && r.residue_seq_code === filterNum)
      );
    }

    // Sort
    const sorted = [...filtered].sort((a, b) => {
      let comparison = 0;
      switch (sortField) {
        case 'residue':
          comparison = a.residue_seq_code - b.residue_seq_code;
          break;
        case 'atom':
          comparison = a.atom_name.localeCompare(b.atom_name);
          break;
        case 'shift':
          comparison = a.assigned_shift - b.assigned_shift;
          break;
        case 'confidence':
          comparison = a.confidence - b.confidence;
          break;
      }
      return sortDirection === 'asc' ? comparison : -comparison;
    });

    return sorted;
  }, [assignmentResults, sortField, sortDirection, filter]);

  const handleSort = (field: SortField) => {
    if (field === sortField) {
      setSortDirection((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDirection(field === 'confidence' ? 'desc' : 'asc');
    }
  };

  const SortIcon = ({ field }: { field: SortField }) => {
    if (field !== sortField) return <span className="text-slate-600">&#8645;</span>;
    return sortDirection === 'asc' ? (
      <span className="text-blue-400">&#8593;</span>
    ) : (
      <span className="text-blue-400">&#8595;</span>
    );
  };

  // Stats
  const totalAssigned = assignmentResults.length;
  const avgConfidence =
    totalAssigned > 0
      ? assignmentResults.reduce((sum, r) => sum + r.confidence, 0) / totalAssigned
      : 0;

  if (assignmentResults.length === 0) {
    return (
      <div className="panel h-full flex items-center justify-center text-slate-500 text-sm">
        <div className="text-center">
          <p>No assignment results</p>
          <p className="text-xs mt-1">Click "Run Assignment" to compute assignments</p>
        </div>
      </div>
    );
  }

  return (
    <div className="panel h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-2 border-b border-slate-700">
        <div className="text-sm">
          <span className="font-medium text-slate-200">Assignments</span>
          <span className="text-slate-400 ml-2">
            ({totalAssigned} | avg: {(avgConfidence * 100).toFixed(0)}%)
          </span>
        </div>
        <input
          type="text"
          placeholder="Filter..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="input text-xs w-28"
        />
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-slate-800">
            <tr className="text-left text-slate-400 text-xs">
              <th
                className="p-2 cursor-pointer hover:text-slate-200"
                onClick={() => handleSort('residue')}
              >
                Residue <SortIcon field="residue" />
              </th>
              <th
                className="p-2 cursor-pointer hover:text-slate-200"
                onClick={() => handleSort('atom')}
              >
                Atom <SortIcon field="atom" />
              </th>
              <th
                className="p-2 cursor-pointer hover:text-slate-200"
                onClick={() => handleSort('shift')}
              >
                Shift <SortIcon field="shift" />
              </th>
              <th
                className="p-2 cursor-pointer hover:text-slate-200"
                onClick={() => handleSort('confidence')}
              >
                Conf <SortIcon field="confidence" />
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedResults.map((result) => (
              <AssignmentRow
                key={result.atom_id}
                result={result}
                isSelected={result.atom_id === selectedAtomId}
                isResidueSelected={result.residue_seq_code === selectedResidueSeqCode}
                confidenceLevel={getConfidenceLevel(result.confidence)}
                showAlternatives={showAlternatives === result.atom_id}
                onSelect={() => {
                  selectAtom(result.atom_id);
                  selectResidue(result.residue_seq_code);
                }}
                onToggleAlternatives={() =>
                  setShowAlternatives(
                    showAlternatives === result.atom_id ? null : result.atom_id
                  )
                }
              />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

interface AssignmentRowProps {
  result: AssignmentResult;
  isSelected: boolean;
  isResidueSelected: boolean;
  confidenceLevel: ConfidenceLevel;
  showAlternatives: boolean;
  onSelect: () => void;
  onToggleAlternatives: () => void;
}

function AssignmentRow({
  result,
  isSelected,
  isResidueSelected,
  confidenceLevel,
  showAlternatives,
  onSelect,
  onToggleAlternatives,
}: AssignmentRowProps) {
  const hasAlternatives = result.alternative_shifts.length > 0;

  return (
    <>
      <tr
        className={`cursor-pointer hover:bg-slate-700/50 ${
          isSelected
            ? 'bg-blue-900/40'
            : isResidueSelected
            ? 'bg-slate-700/30'
            : ''
        }`}
        onClick={onSelect}
      >
        <td className="p-2">
          <span className="text-slate-400">{result.residue_seq_code}</span>
          <span className="text-slate-300 ml-1">{result.residue_name}</span>
        </td>
        <td className="p-2 font-mono text-slate-200">{result.atom_name}</td>
        <td className="p-2 font-mono text-slate-200">
          {result.assigned_shift.toFixed(2)}
        </td>
        <td className="p-2">
          <div className="flex items-center gap-1">
            <span
              className={`px-1.5 py-0.5 rounded text-xs font-medium ${CONFIDENCE_COLORS[confidenceLevel]}`}
            >
              {(result.confidence * 100).toFixed(0)}%
            </span>
            {hasAlternatives && (
              <button
                className="text-slate-500 hover:text-slate-300 text-xs"
                onClick={(e) => {
                  e.stopPropagation();
                  onToggleAlternatives();
                }}
                title="Show alternatives"
              >
                +{result.alternative_shifts.length}
              </button>
            )}
          </div>
        </td>
      </tr>
      {showAlternatives && hasAlternatives && (
        <tr className="bg-slate-800/50">
          <td colSpan={4} className="p-2 pl-8">
            <div className="text-xs text-slate-400">
              <span className="font-medium">Alternatives:</span>
              <div className="flex flex-wrap gap-2 mt-1">
                {result.alternative_shifts.map(([shift, prob], i) => (
                  <span
                    key={i}
                    className="px-2 py-0.5 bg-slate-700 rounded text-slate-300"
                  >
                    {shift.toFixed(2)} ({(prob * 100).toFixed(0)}%)
                  </span>
                ))}
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}
