/**
 * Canvas-based visualization of a single spin system.
 * Shows atoms as nodes and correlations as edges.
 */

import { useRef, useEffect, useCallback, useState } from 'react';
import { useAssignmentStore } from '../../stores/assignmentStore';
import type { SpinSystemAtom, ConfidenceLevel } from '../../types/tauri';

// Colors for the dark theme
const COLORS = {
  background: '#1e293b', // slate-800
  atomHigh: '#22c55e', // green-500
  atomMedium: '#f59e0b', // amber-500
  atomLow: '#ef4444', // red-500
  atomUnassigned: '#64748b', // slate-500
  correlation: '#475569', // slate-600
  correlationStrong: '#94a3b8', // slate-400
  selected: '#3b82f6', // blue-500
  label: '#e2e8f0', // slate-200
  labelDim: '#94a3b8', // slate-400
};

// Fixed positions for backbone atoms (relative coordinates 0-1)
const ATOM_POSITIONS: Record<string, { x: number; y: number }> = {
  H: { x: 0.1, y: 0.35 },
  N: { x: 0.25, y: 0.35 },
  CA: { x: 0.45, y: 0.35 },
  HA: { x: 0.45, y: 0.15 },
  C: { x: 0.65, y: 0.35 },
  O: { x: 0.65, y: 0.15 },
  CB: { x: 0.45, y: 0.55 },
};

function getConfidenceColor(level: ConfidenceLevel): string {
  switch (level) {
    case 'high':
      return COLORS.atomHigh;
    case 'medium':
      return COLORS.atomMedium;
    case 'low':
      return COLORS.atomLow;
    default:
      return COLORS.atomUnassigned;
  }
}

export function SpinSystemView() {
  const {
    selectedResidueSeqCode,
    spinSystems,
    selectedAtomId,
    selectAtom,
    getConfidenceLevel,
  } = useAssignmentStore();

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredAtom, setHoveredAtom] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    atom: SpinSystemAtom;
  } | null>(null);

  const spinSystem =
    selectedResidueSeqCode !== null
      ? spinSystems.get(selectedResidueSeqCode)
      : null;

  // Convert relative position to canvas coordinates
  const relativeToCanvas = useCallback(
    (relX: number, relY: number, width: number, height: number) => {
      const padding = 40;
      const plotWidth = width - padding * 2;
      const plotHeight = height - padding * 2;
      return {
        x: padding + relX * plotWidth,
        y: padding + relY * plotHeight,
      };
    },
    []
  );

  // Find atom at position
  const findAtomAtPosition = useCallback(
    (canvasX: number, canvasY: number, width: number, height: number) => {
      if (!spinSystem) return null;

      const nodeRadius = 18;

      for (const atom of spinSystem.atoms) {
        const pos = ATOM_POSITIONS[atom.atom_name];
        if (!pos) continue;

        const { x, y } = relativeToCanvas(pos.x, pos.y, width, height);
        const dist = Math.sqrt((canvasX - x) ** 2 + (canvasY - y) ** 2);

        if (dist <= nodeRadius) {
          return atom;
        }
      }

      return null;
    },
    [spinSystem, relativeToCanvas]
  );

  // Render function
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle DPI scaling
    const dpr = window.devicePixelRatio || 1;
    const rect = container.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.scale(dpr, dpr);

    // Clear background
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, width, height);

    if (!spinSystem) {
      // No spin system selected
      ctx.fillStyle = COLORS.labelDim;
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Select a residue to view spin system', width / 2, height / 2);
      return;
    }

    // Draw title
    ctx.fillStyle = COLORS.label;
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(
      `${spinSystem.residue_name} ${spinSystem.residue_seq_code}`,
      10,
      20
    );

    // Draw overall confidence
    const overallLevel = getConfidenceLevel(spinSystem.overall_confidence);
    ctx.fillStyle = getConfidenceColor(overallLevel);
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(
      `${(spinSystem.overall_confidence * 100).toFixed(0)}% avg`,
      width - 10,
      20
    );

    // Draw correlations (edges)
    ctx.strokeStyle = COLORS.correlationStrong;
    ctx.lineWidth = 2;

    for (const corr of spinSystem.correlations) {
      const fromPos = ATOM_POSITIONS[corr.from_atom];
      const toPos = ATOM_POSITIONS[corr.to_atom];
      if (!fromPos || !toPos) continue;

      const from = relativeToCanvas(fromPos.x, fromPos.y, width, height);
      const to = relativeToCanvas(toPos.x, toPos.y, width, height);

      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
    }

    // Draw atoms (nodes)
    const nodeRadius = 18;

    for (const atom of spinSystem.atoms) {
      const pos = ATOM_POSITIONS[atom.atom_name];
      if (!pos) continue;

      const { x, y } = relativeToCanvas(pos.x, pos.y, width, height);
      const level = getConfidenceLevel(atom.confidence);
      const color = getConfidenceColor(level);
      const isSelected = atom.atom_id === selectedAtomId;
      const isHovered = atom.atom_id === hoveredAtom;

      // Selection ring
      if (isSelected) {
        ctx.beginPath();
        ctx.arc(x, y, nodeRadius + 4, 0, Math.PI * 2);
        ctx.strokeStyle = COLORS.selected;
        ctx.lineWidth = 3;
        ctx.stroke();
      }

      // Hover ring
      if (isHovered && !isSelected) {
        ctx.beginPath();
        ctx.arc(x, y, nodeRadius + 2, 0, Math.PI * 2);
        ctx.strokeStyle = '#6b7280';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Node fill
      ctx.beginPath();
      ctx.arc(x, y, nodeRadius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      // Node border
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Atom label
      ctx.fillStyle = '#1e293b';
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(atom.atom_name, x, y);

      // Shift value below
      if (atom.assigned_shift !== null) {
        ctx.fillStyle = COLORS.labelDim;
        ctx.font = '10px monospace';
        ctx.fillText(atom.assigned_shift.toFixed(1), x, y + nodeRadius + 12);
      }
    }

    // Draw legend
    const legendY = height - 20;
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';

    const legendItems = [
      { color: COLORS.atomHigh, label: 'High' },
      { color: COLORS.atomMedium, label: 'Med' },
      { color: COLORS.atomLow, label: 'Low' },
    ];

    let legendX = 10;
    for (const item of legendItems) {
      ctx.fillStyle = item.color;
      ctx.fillRect(legendX, legendY - 8, 12, 12);
      ctx.fillStyle = COLORS.labelDim;
      ctx.fillText(item.label, legendX + 16, legendY);
      legendX += 50;
    }
  }, [
    spinSystem,
    selectedAtomId,
    hoveredAtom,
    relativeToCanvas,
    getConfidenceLevel,
  ]);

  // Handle mouse move
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      const container = containerRef.current;
      if (!canvas || !container) return;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const width = rect.width;
      const height = rect.height;

      const atom = findAtomAtPosition(x, y, width, height);

      if (atom) {
        setHoveredAtom(atom.atom_id);
        const pos = ATOM_POSITIONS[atom.atom_name];
        if (pos) {
          const { x: tooltipX, y: tooltipY } = relativeToCanvas(
            pos.x,
            pos.y,
            width,
            height
          );
          setTooltip({ x: tooltipX, y: tooltipY - 40, atom });
        }
      } else {
        setHoveredAtom(null);
        setTooltip(null);
      }
    },
    [findAtomAtPosition, relativeToCanvas]
  );

  // Handle click
  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const width = rect.width;
      const height = rect.height;

      const atom = findAtomAtPosition(x, y, width, height);

      if (atom) {
        selectAtom(atom.atom_id);
      }
    },
    [findAtomAtPosition, selectAtom]
  );

  // Handle mouse leave
  const handleMouseLeave = useCallback(() => {
    setHoveredAtom(null);
    setTooltip(null);
  }, []);

  // Re-render on changes
  useEffect(() => {
    render();
  }, [render]);

  // Handle resize
  useEffect(() => {
    const handleResize = () => render();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [render]);

  return (
    <div ref={containerRef} className="relative w-full h-full min-h-[200px]">
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-pointer"
        onMouseMove={handleMouseMove}
        onClick={handleClick}
        onMouseLeave={handleMouseLeave}
      />

      {/* Tooltip */}
      {tooltip && (
        <div
          className="absolute pointer-events-none bg-slate-900 border border-slate-600 rounded px-2 py-1 text-xs shadow-lg"
          style={{
            left: tooltip.x,
            top: tooltip.y,
            transform: 'translate(-50%, -100%)',
          }}
        >
          <div className="font-medium text-slate-200">
            {tooltip.atom.atom_name}
          </div>
          {tooltip.atom.assigned_shift !== null && (
            <div className="text-slate-400">
              {tooltip.atom.assigned_shift.toFixed(2)} ppm
            </div>
          )}
          <div className="text-slate-400">
            {(tooltip.atom.confidence * 100).toFixed(0)}% confidence
          </div>
          {tooltip.atom.alternatives.length > 0 && (
            <div className="text-slate-500 text-[10px] mt-0.5">
              +{tooltip.atom.alternatives.length} alternatives
            </div>
          )}
        </div>
      )}
    </div>
  );
}
