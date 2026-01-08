/**
 * Horizontal sequence viewer with residue coloring by assignment status.
 */

import { useAssignmentStore } from '../../stores/assignmentStore';
import type { AssignmentStatus } from '../../types/tauri';

const STATUS_COLORS: Record<AssignmentStatus, string> = {
  unassigned: 'bg-slate-700 hover:bg-slate-600',
  partial: 'bg-amber-700 hover:bg-amber-600',
  complete: 'bg-green-700 hover:bg-green-600',
};

export function SequenceViewer() {
  const {
    molecule,
    residues,
    selectedResidueSeqCode,
    selectResidue,
    getAssignmentStatus,
  } = useAssignmentStore();

  if (!molecule) {
    return (
      <div className="panel p-4 text-slate-500 text-sm">
        No molecule loaded. Load a sequence to view residues.
      </div>
    );
  }

  return (
    <div className="panel">
      {/* Header */}
      <div className="flex items-center justify-between p-2 border-b border-slate-700">
        <h3 className="text-sm font-medium text-slate-200">
          {molecule.name}
          <span className="ml-2 text-slate-400">
            ({residues.length} residues)
          </span>
        </h3>

        {/* Legend */}
        <div className="flex items-center gap-3 text-xs text-slate-400">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-slate-700" />
            Unassigned
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-amber-700" />
            Partial
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-green-700" />
            Complete
          </span>
        </div>
      </div>

      {/* Sequence */}
      <div className="p-2 overflow-x-auto">
        <div className="flex flex-wrap gap-0.5">
          {residues.map((residue) => {
            const status = getAssignmentStatus(residue.sequence_code);
            const isSelected = residue.sequence_code === selectedResidueSeqCode;

            return (
              <button
                key={residue.id}
                onClick={() => selectResidue(residue.sequence_code)}
                className={`
                  w-7 h-9 flex flex-col items-center justify-center rounded text-xs
                  transition-colors cursor-pointer
                  ${STATUS_COLORS[status]}
                  ${isSelected ? 'ring-2 ring-blue-500 ring-offset-1 ring-offset-slate-800' : ''}
                `}
                title={`${residue.residue_name} ${residue.sequence_code}`}
              >
                <span className="font-mono font-medium text-slate-100">
                  {residue.one_letter_code}
                </span>
                <span className="text-[9px] text-slate-400">
                  {residue.sequence_code}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
