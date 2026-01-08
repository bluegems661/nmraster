/**
 * Canvas-based 2D spectrum renderer with contour plots.
 * Uses marching squares algorithm for contour extraction.
 */

import { useRef, useEffect, useCallback, useMemo, useState } from 'react';
import type { Spectrum2DDataResponse } from '../../types/tauri';

interface ViewBounds2D {
  xMin: number; // F2 (direct dimension, typically 1H) - low ppm on right
  xMax: number; // F2 high ppm on left
  yMin: number; // F1 (indirect dimension) - low ppm at TOP (inverted)
  yMax: number; // F1 high ppm at BOTTOM (inverted)
}

interface SpectrumCanvas2DProps {
  spectrum: Spectrum2DDataResponse | null;
  viewBounds: ViewBounds2D | null;
  contourLevels?: number;
  contourBase?: number; // multiplier of noise floor
  showPositive?: boolean;
  showNegative?: boolean;
  onViewChange?: (bounds: ViewBounds2D) => void;
  onMouseMove?: (f2Ppm: number, f1Ppm: number, intensity: number) => void;
}

// Colors for the dark theme
const COLORS = {
  background: '#1e293b', // slate-800
  grid: '#334155', // slate-700
  axis: '#94a3b8', // slate-400
  contourPositive: [
    '#93c5fd', // blue-300
    '#60a5fa', // blue-400
    '#3b82f6', // blue-500
    '#2563eb', // blue-600
    '#1d4ed8', // blue-700
    '#1e40af', // blue-800
  ],
  contourNegative: [
    '#fca5a5', // red-300
    '#f87171', // red-400
    '#ef4444', // red-500
    '#dc2626', // red-600
    '#b91c1c', // red-700
    '#991b1b', // red-800
  ],
  crosshair: '#fbbf24', // amber-400
  zoomBox: 'rgba(251, 191, 36, 0.3)', // amber with transparency
  zoomBoxBorder: '#fbbf24',
};

export function SpectrumCanvas2D({
  spectrum,
  viewBounds,
  contourLevels = 8,
  contourBase = 5,
  showPositive = true,
  showNegative = true,
  onViewChange,
  onMouseMove,
}: SpectrumCanvas2DProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Interaction state
  const [zoomBox, setZoomBox] = useState<{ startX: number; startY: number; endX: number; endY: number } | null>(null);
  const isDragging = useRef(false);
  const dragStart = useRef({ x: 0, y: 0 });
  const mousePos = useRef<{ x: number; y: number } | null>(null);

  // Plot area dimensions (consistent across render)
  const getPlotArea = useCallback((width: number, height: number) => ({
    left: 70,
    right: width - 20,
    top: 20,
    bottom: height - 50,
    width: width - 90,
    height: height - 70,
  }), []);

  // Calculate contour threshold levels
  const levels = useMemo(() => {
    if (!spectrum) return { positive: [], negative: [] };

    // Find max value in data
    let maxVal = 0;
    for (const row of spectrum.data) {
      for (const val of row) {
        if (Math.abs(val) > maxVal) maxVal = Math.abs(val);
      }
    }

    // Use noise_floor if available, otherwise derive from max value
    const effectiveNoiseFloor = spectrum.noise_floor > 0
      ? spectrum.noise_floor
      : maxVal / 100;

    const baseLevel = effectiveNoiseFloor * contourBase;
    const positive: number[] = [];
    const negative: number[] = [];

    let level = baseLevel;
    for (let i = 0; i < contourLevels; i++) {
      positive.push(level);
      negative.push(-level);
      level *= 1.4;
    }

    return { positive, negative };
  }, [spectrum, contourLevels, contourBase]);

  // Convert canvas coordinates to spectrum coordinates
  // F1 is INVERTED: low PPM at top, high PPM at bottom
  const canvasToSpectrum = useCallback(
    (canvasX: number, canvasY: number, width: number, height: number) => {
      if (!viewBounds) return { f2Ppm: 0, f1Ppm: 0 };

      const plot = getPlotArea(width, height);

      // F2 (x-axis): high ppm on left, low on right
      const f2Ppm = viewBounds.xMax - ((canvasX - plot.left) / plot.width) * (viewBounds.xMax - viewBounds.xMin);
      // F1 (y-axis): INVERTED - low ppm at top, high ppm at bottom
      const f1Ppm = viewBounds.yMin + ((canvasY - plot.top) / plot.height) * (viewBounds.yMax - viewBounds.yMin);

      return { f2Ppm, f1Ppm };
    },
    [viewBounds, getPlotArea]
  );

  // Get intensity at spectrum coordinates
  const getIntensityAt = useCallback(
    (f2Ppm: number, f1Ppm: number): number => {
      if (!spectrum) return 0;

      const { data, ppm_axis_f1, ppm_axis_f2 } = spectrum;

      // Find indices (PPM axes are descending: high to low)
      let f1Idx = 0;
      for (let i = 0; i < ppm_axis_f1.length - 1; i++) {
        if (f1Ppm <= ppm_axis_f1[i] && f1Ppm > ppm_axis_f1[i + 1]) {
          f1Idx = i;
          break;
        }
      }

      let f2Idx = 0;
      for (let i = 0; i < ppm_axis_f2.length - 1; i++) {
        if (f2Ppm <= ppm_axis_f2[i] && f2Ppm > ppm_axis_f2[i + 1]) {
          f2Idx = i;
          break;
        }
      }

      const f2Length = data[0]?.length ?? 1;
      f1Idx = Math.max(0, Math.min(f1Idx, data.length - 1));
      f2Idx = Math.max(0, Math.min(f2Idx, f2Length - 1));

      return data[f1Idx]?.[f2Idx] ?? 0;
    },
    [spectrum]
  );

  // Draw contours using marching squares with proper interpolation
  const drawContours = useCallback(
    (ctx: CanvasRenderingContext2D, width: number, height: number) => {
      if (!spectrum || !viewBounds) return;

      const { data, ppm_axis_f1, ppm_axis_f2 } = spectrum;
      const plot = getPlotArea(width, height);

      // Determine visible range in data indices
      // PPM axes are descending (high to low)
      let f1StartIdx = 0, f1EndIdx = data.length;
      let f2StartIdx = 0, f2EndIdx = data[0]?.length ?? 0;

      for (let i = 0; i < ppm_axis_f1.length; i++) {
        if (ppm_axis_f1[i] <= viewBounds.yMax && f1StartIdx === 0) f1StartIdx = i;
        if (ppm_axis_f1[i] < viewBounds.yMin) { f1EndIdx = i; break; }
      }
      for (let i = 0; i < ppm_axis_f2.length; i++) {
        if (ppm_axis_f2[i] <= viewBounds.xMax && f2StartIdx === 0) f2StartIdx = i;
        if (ppm_axis_f2[i] < viewBounds.xMin) { f2EndIdx = i; break; }
      }

      // Calculate step size for performance
      const targetCells = 300;
      const f1Step = Math.max(1, Math.floor((f1EndIdx - f1StartIdx) / targetCells));
      const f2Step = Math.max(1, Math.floor((f2EndIdx - f2StartIdx) / targetCells));

      // Helper to convert data index to canvas position
      const idxToCanvas = (f1Idx: number, f2Idx: number) => {
        const f1Ppm = ppm_axis_f1[f1Idx];
        const f2Ppm = ppm_axis_f2[f2Idx];
        // F2: high ppm left, F1: INVERTED (low ppm top)
        const x = plot.left + ((viewBounds.xMax - f2Ppm) / (viewBounds.xMax - viewBounds.xMin)) * plot.width;
        const y = plot.top + ((f1Ppm - viewBounds.yMin) / (viewBounds.yMax - viewBounds.yMin)) * plot.height;
        return { x, y };
      };

      // Draw contour levels from highest to lowest so lowest (strongest) are on top
      const allLevels: { level: number; color: string; isNegative: boolean }[] = [];

      if (showPositive) {
        levels.positive.forEach((level, i) => {
          allLevels.push({
            level,
            color: COLORS.contourPositive[Math.min(i, COLORS.contourPositive.length - 1)],
            isNegative: false
          });
        });
      }

      if (showNegative) {
        levels.negative.forEach((level, i) => {
          allLevels.push({
            level: Math.abs(level),
            color: COLORS.contourNegative[Math.min(i, COLORS.contourNegative.length - 1)],
            isNegative: true
          });
        });
      }

      // Sort by level descending (draw weak contours first)
      allLevels.sort((a, b) => b.level - a.level);

      // Marching squares: for each case, which edges to connect
      // Edge indices: 0=left (tl→bl), 1=bottom (bl→br), 2=right (br→tr), 3=top (tl→tr)
      // Bits: bl=1, br=2, tr=4, tl=8
      const edgePairs: Record<number, [number, number][]> = {
        0: [],                    // all below threshold
        1: [[0, 1]],              // bl above → left to bottom
        2: [[1, 2]],              // br above → bottom to right
        3: [[0, 2]],              // bl,br above → left to right
        4: [[2, 3]],              // tr above → right to top
        5: [[0, 3], [1, 2]],      // bl,tr above (saddle) → two lines
        6: [[1, 3]],              // br,tr above → bottom to top
        7: [[0, 3]],              // bl,br,tr above (only tl below) → left to top
        8: [[0, 3]],              // tl above → left to top
        9: [[1, 3]],              // tl,bl above → bottom to top
        10: [[0, 1], [2, 3]],     // tl,br above (saddle) → two lines
        11: [[2, 3]],             // tl,bl,br above (only tr below) → right to top
        12: [[0, 2]],             // tl,tr above → left to right
        13: [[1, 2]],             // tl,bl,tr above (only br below) → bottom to right
        14: [[0, 1]],             // tl,br,tr above (only bl below) → left to bottom
        15: [],                   // all above threshold
      };

      // Interpolate along an edge to find where the contour crosses
      const interpolateEdge = (
        edge: number,
        tl: { x: number; y: number },
        tr: { x: number; y: number },
        bl: { x: number; y: number },
        br: { x: number; y: number },
        v_tl: number,
        v_tr: number,
        v_bl: number,
        v_br: number,
        level: number
      ): { x: number; y: number } => {
        let p0: { x: number; y: number }, p1: { x: number; y: number };
        let v0: number, v1: number;

        switch (edge) {
          case 0: // left: tl to bl
            p0 = tl; p1 = bl; v0 = v_tl; v1 = v_bl;
            break;
          case 1: // bottom: bl to br
            p0 = bl; p1 = br; v0 = v_bl; v1 = v_br;
            break;
          case 2: // right: br to tr
            p0 = br; p1 = tr; v0 = v_br; v1 = v_tr;
            break;
          case 3: // top: tl to tr
            p0 = tl; p1 = tr; v0 = v_tl; v1 = v_tr;
            break;
          default:
            return tl;
        }

        // Calculate interpolation factor: where does level cross between v0 and v1?
        let t: number;
        if (Math.abs(v1 - v0) < 1e-10) {
          t = 0.5; // Avoid division by zero
        } else {
          t = (level - v0) / (v1 - v0);
          t = Math.max(0, Math.min(1, t)); // Clamp to [0,1]
        }

        return {
          x: p0.x + t * (p1.x - p0.x),
          y: p0.y + t * (p1.y - p0.y),
        };
      };

      for (const { level, color, isNegative } of allLevels) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.beginPath();

        for (let i = f1StartIdx; i < f1EndIdx - f1Step; i += f1Step) {
          for (let j = f2StartIdx; j < f2EndIdx - f2Step; j += f2Step) {
            // Get corner values (adjust for negative contours)
            // Cell layout (in canvas space after coordinate transform):
            //   tl (i, j)      ----  tr (i, j+step)
            //      |                    |
            //   bl (i+step, j) ----  br (i+step, j+step)
            let v_tl = data[i]?.[j] ?? 0;
            let v_tr = data[i]?.[j + f2Step] ?? 0;
            let v_bl = data[i + f1Step]?.[j] ?? 0;
            let v_br = data[i + f1Step]?.[j + f2Step] ?? 0;

            if (isNegative) {
              v_tl = -v_tl; v_tr = -v_tr; v_bl = -v_bl; v_br = -v_br;
            }

            // Calculate marching squares index based on which corners are above threshold
            let idx = 0;
            if (v_bl >= level) idx |= 1;
            if (v_br >= level) idx |= 2;
            if (v_tr >= level) idx |= 4;
            if (v_tl >= level) idx |= 8;

            const pairs = edgePairs[idx];
            if (!pairs || pairs.length === 0) continue;

            // Get corner canvas positions
            const tl = idxToCanvas(i, j);
            const tr = idxToCanvas(i, j + f2Step);
            const bl = idxToCanvas(i + f1Step, j);
            const br = idxToCanvas(i + f1Step, j + f2Step);

            // Draw line segments between edge crossing points
            for (const [edge1, edge2] of pairs) {
              const p1 = interpolateEdge(edge1, tl, tr, bl, br, v_tl, v_tr, v_bl, v_br, level);
              const p2 = interpolateEdge(edge2, tl, tr, bl, br, v_tl, v_tr, v_bl, v_br, level);

              ctx.moveTo(p1.x, p1.y);
              ctx.lineTo(p2.x, p2.y);
            }
          }
        }

        ctx.stroke();
      }
    },
    [spectrum, viewBounds, levels, showPositive, showNegative, getPlotArea]
  );

  // Main render function
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = container.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.scale(dpr, dpr);

    // Clear
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, width, height);

    if (!spectrum || !viewBounds) {
      ctx.fillStyle = COLORS.axis;
      ctx.font = '14px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText('No 2D spectrum loaded', width / 2, height / 2);
      return;
    }

    const plot = getPlotArea(width, height);

    // Draw grid
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 0.5;

    // F2 grid (x-axis)
    const f2Range = viewBounds.xMax - viewBounds.xMin;
    const f2Step = calculateNiceStep(f2Range, 8);
    const f2Start = Math.ceil(viewBounds.xMin / f2Step) * f2Step;

    for (let ppm = f2Start; ppm <= viewBounds.xMax; ppm += f2Step) {
      const x = plot.left + ((viewBounds.xMax - ppm) / f2Range) * plot.width;
      ctx.beginPath();
      ctx.moveTo(x, plot.top);
      ctx.lineTo(x, plot.bottom);
      ctx.stroke();
    }

    // F1 grid (y-axis) - INVERTED
    const f1Range = viewBounds.yMax - viewBounds.yMin;
    const f1Step = calculateNiceStep(f1Range, 6);
    const f1Start = Math.ceil(viewBounds.yMin / f1Step) * f1Step;

    for (let ppm = f1Start; ppm <= viewBounds.yMax; ppm += f1Step) {
      // INVERTED: low ppm at top, high at bottom
      const y = plot.top + ((ppm - viewBounds.yMin) / f1Range) * plot.height;
      ctx.beginPath();
      ctx.moveTo(plot.left, y);
      ctx.lineTo(plot.right, y);
      ctx.stroke();
    }

    // Draw contours
    drawContours(ctx, width, height);

    // Draw axes
    ctx.strokeStyle = COLORS.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plot.left, plot.top);
    ctx.lineTo(plot.left, plot.bottom);
    ctx.lineTo(plot.right, plot.bottom);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = COLORS.axis;
    ctx.font = '11px system-ui';

    // F2 labels (bottom)
    ctx.textAlign = 'center';
    for (let ppm = f2Start; ppm <= viewBounds.xMax; ppm += f2Step) {
      const x = plot.left + ((viewBounds.xMax - ppm) / f2Range) * plot.width;
      ctx.fillText(ppm.toFixed(1), x, plot.bottom + 15);
    }

    // F1 labels (left) - INVERTED
    ctx.textAlign = 'right';
    for (let ppm = f1Start; ppm <= viewBounds.yMax; ppm += f1Step) {
      const y = plot.top + ((ppm - viewBounds.yMin) / f1Range) * plot.height;
      ctx.fillText(ppm.toFixed(1), plot.left - 5, y + 4);
    }

    // Axis titles
    ctx.font = '12px system-ui';
    ctx.textAlign = 'center';

    const f2Label = spectrum.experiment_type.includes('HSQC') ? '¹H (ppm)' : 'F2 (ppm)';
    ctx.fillText(f2Label, plot.left + plot.width / 2, height - 5);

    ctx.save();
    ctx.translate(15, plot.top + plot.height / 2);
    ctx.rotate(-Math.PI / 2);
    const f1Label = spectrum.experiment_type.includes('HSQC')
      ? (spectrum.experiment_type.includes('13C') ? '¹³C (ppm)' : '¹⁵N (ppm)')
      : 'F1 (ppm)';
    ctx.fillText(f1Label, 0, 0);
    ctx.restore();

    // Draw crosshair
    if (mousePos.current && !zoomBox) {
      const { x, y } = mousePos.current;
      if (x >= plot.left && x <= plot.right && y >= plot.top && y <= plot.bottom) {
        ctx.strokeStyle = COLORS.crosshair;
        ctx.lineWidth = 0.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(x, plot.top);
        ctx.lineTo(x, plot.bottom);
        ctx.moveTo(plot.left, y);
        ctx.lineTo(plot.right, y);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    // Draw zoom box
    if (zoomBox) {
      ctx.fillStyle = COLORS.zoomBox;
      ctx.strokeStyle = COLORS.zoomBoxBorder;
      ctx.lineWidth = 1;
      const x = Math.min(zoomBox.startX, zoomBox.endX);
      const y = Math.min(zoomBox.startY, zoomBox.endY);
      const w = Math.abs(zoomBox.endX - zoomBox.startX);
      const h = Math.abs(zoomBox.endY - zoomBox.startY);
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
    }
  }, [spectrum, viewBounds, drawContours, zoomBox, getPlotArea]);

  useEffect(() => {
    const handleResize = () => render();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [render]);

  useEffect(() => {
    render();
  }, [render]);

  // Mouse handlers for box zoom
  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!viewBounds) return;
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      isDragging.current = true;
      dragStart.current = { x, y };
      setZoomBox({ startX: x, startY: y, endX: x, endY: y });
    },
    [viewBounds]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas || !viewBounds) return;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      mousePos.current = { x, y };

      // Report coordinates
      const coords = canvasToSpectrum(x, y, rect.width, rect.height);
      const intensity = getIntensityAt(coords.f2Ppm, coords.f1Ppm);
      onMouseMove?.(coords.f2Ppm, coords.f1Ppm, intensity);

      // Update zoom box
      if (isDragging.current && zoomBox) {
        setZoomBox({ ...zoomBox, endX: x, endY: y });
      }

      render();
    },
    [viewBounds, canvasToSpectrum, getIntensityAt, onMouseMove, zoomBox, render]
  );

  const handleMouseUp = useCallback(
    (_e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!isDragging.current || !zoomBox || !viewBounds || !onViewChange) {
        isDragging.current = false;
        setZoomBox(null);
        return;
      }

      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;

      // Only zoom if box is large enough
      const boxWidth = Math.abs(zoomBox.endX - zoomBox.startX);
      const boxHeight = Math.abs(zoomBox.endY - zoomBox.startY);

      if (boxWidth > 10 && boxHeight > 10) {
        const start = canvasToSpectrum(
          Math.min(zoomBox.startX, zoomBox.endX),
          Math.min(zoomBox.startY, zoomBox.endY),
          width, height
        );
        const end = canvasToSpectrum(
          Math.max(zoomBox.startX, zoomBox.endX),
          Math.max(zoomBox.startY, zoomBox.endY),
          width, height
        );

        onViewChange({
          xMin: Math.min(start.f2Ppm, end.f2Ppm),
          xMax: Math.max(start.f2Ppm, end.f2Ppm),
          yMin: Math.min(start.f1Ppm, end.f1Ppm),
          yMax: Math.max(start.f1Ppm, end.f1Ppm),
        });
      }

      isDragging.current = false;
      setZoomBox(null);
    },
    [zoomBox, viewBounds, onViewChange, canvasToSpectrum]
  );

  const handleMouseLeave = useCallback(() => {
    isDragging.current = false;
    setZoomBox(null);
    mousePos.current = null;
    render();
  }, [render]);

  // Scroll to zoom
  const handleWheel = useCallback(
    (e: React.WheelEvent<HTMLCanvasElement>) => {
      if (!onViewChange || !viewBounds) return;
      e.preventDefault();

      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const center = canvasToSpectrum(e.clientX - rect.left, e.clientY - rect.top, rect.width, rect.height);

      const factor = e.deltaY > 0 ? 1.2 : 0.8;

      onViewChange({
        xMin: center.f2Ppm - (center.f2Ppm - viewBounds.xMin) * factor,
        xMax: center.f2Ppm + (viewBounds.xMax - center.f2Ppm) * factor,
        yMin: center.f1Ppm - (center.f1Ppm - viewBounds.yMin) * factor,
        yMax: center.f1Ppm + (viewBounds.yMax - center.f1Ppm) * factor,
      });
    },
    [viewBounds, canvasToSpectrum, onViewChange]
  );

  return (
    <div ref={containerRef} className="w-full h-full min-h-[400px]">
      <canvas
        ref={canvasRef}
        className="cursor-crosshair"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onWheel={handleWheel}
      />
    </div>
  );
}

function calculateNiceStep(range: number, targetTicks: number): number {
  const rawStep = range / targetTicks;
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const normalized = rawStep / magnitude;

  let niceNormalized: number;
  if (normalized <= 1) niceNormalized = 1;
  else if (normalized <= 2) niceNormalized = 2;
  else if (normalized <= 5) niceNormalized = 5;
  else niceNormalized = 10;

  return niceNormalized * magnitude;
}

export type { ViewBounds2D };
