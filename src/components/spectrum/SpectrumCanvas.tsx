/**
 * Canvas-based 1D spectrum renderer.
 * Handles drawing the spectrum trace, peaks, and axis labels.
 */

import { useRef, useEffect, useCallback } from 'react';
import type { SpectrumDataResponse, PickedPeak, ViewBounds } from '../../types/tauri';

interface SpectrumCanvasProps {
  spectrum: SpectrumDataResponse | null;
  peaks: PickedPeak[];
  viewBounds: ViewBounds | null;
  selectedPeakId: string | null;
  lineWidth?: number;
  onPeakSelect?: (peakId: string | null) => void;
  onViewChange?: (bounds: ViewBounds) => void;
  onMouseMove?: (ppm: number, intensity: number) => void;
}

// Colors for the dark theme
const COLORS = {
  background: '#1e293b', // slate-800
  grid: '#334155', // slate-700
  axis: '#94a3b8', // slate-400
  trace: '#22c55e', // green-500
  peak: '#f59e0b', // amber-500
  peakSelected: '#ef4444', // red-500
  peakLabel: '#e2e8f0', // slate-200
};

export function SpectrumCanvas({
  spectrum,
  peaks,
  viewBounds,
  selectedPeakId,
  lineWidth = 1,
  onPeakSelect,
  onViewChange,
  onMouseMove,
}: SpectrumCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Interaction state
  const isDragging = useRef(false);
  const dragStart = useRef({ x: 0, y: 0 });
  const initialBounds = useRef<ViewBounds | null>(null);

  // Convert canvas coordinates to spectrum coordinates
  const canvasToSpectrum = useCallback(
    (canvasX: number, canvasY: number, width: number, height: number) => {
      if (!viewBounds) return { ppm: 0, intensity: 0 };

      const plotLeft = 60;
      const plotRight = width - 20;
      const plotTop = 20;
      const plotBottom = height - 40;
      const plotWidth = plotRight - plotLeft;
      const plotHeight = plotBottom - plotTop;

      // Note: PPM axis is typically reversed (higher values on left)
      const ppm = viewBounds.xMax - ((canvasX - plotLeft) / plotWidth) * (viewBounds.xMax - viewBounds.xMin);
      const intensity = viewBounds.yMax - ((canvasY - plotTop) / plotHeight) * (viewBounds.yMax - viewBounds.yMin);

      return { ppm, intensity };
    },
    [viewBounds]
  );

  // Convert spectrum coordinates to canvas coordinates
  const spectrumToCanvas = useCallback(
    (ppm: number, intensity: number, width: number, height: number) => {
      if (!viewBounds) return { x: 0, y: 0 };

      const plotLeft = 60;
      const plotRight = width - 20;
      const plotTop = 20;
      const plotBottom = height - 40;
      const plotWidth = plotRight - plotLeft;
      const plotHeight = plotBottom - plotTop;

      const x = plotLeft + ((viewBounds.xMax - ppm) / (viewBounds.xMax - viewBounds.xMin)) * plotWidth;
      const y = plotTop + ((viewBounds.yMax - intensity) / (viewBounds.yMax - viewBounds.yMin)) * plotHeight;

      return { x, y };
    },
    [viewBounds]
  );

  // Main render function
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

    // Clear and fill background
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, width, height);

    if (!spectrum || !viewBounds) {
      // Draw placeholder text
      ctx.fillStyle = COLORS.axis;
      ctx.font = '14px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText('No spectrum loaded', width / 2, height / 2);
      return;
    }

    // Plot area dimensions
    const plotLeft = 60;
    const plotRight = width - 20;
    const plotTop = 20;
    const plotBottom = height - 40;
    const plotWidth = plotRight - plotLeft;
    const plotHeight = plotBottom - plotTop;

    // Draw grid
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 0.5;

    // Vertical grid lines (PPM)
    const ppmRange = viewBounds.xMax - viewBounds.xMin;
    const ppmStep = calculateNiceStep(ppmRange, 8);
    const ppmStart = Math.ceil(viewBounds.xMin / ppmStep) * ppmStep;

    for (let ppm = ppmStart; ppm <= viewBounds.xMax; ppm += ppmStep) {
      const x = plotLeft + ((viewBounds.xMax - ppm) / ppmRange) * plotWidth;
      ctx.beginPath();
      ctx.moveTo(x, plotTop);
      ctx.lineTo(x, plotBottom);
      ctx.stroke();
    }

    // Horizontal grid lines (Intensity)
    const intensityRange = viewBounds.yMax - viewBounds.yMin;
    const intensityStep = calculateNiceStep(intensityRange, 6);
    const intensityStart = Math.ceil(viewBounds.yMin / intensityStep) * intensityStep;

    for (let intensity = intensityStart; intensity <= viewBounds.yMax; intensity += intensityStep) {
      const y = plotTop + ((viewBounds.yMax - intensity) / intensityRange) * plotHeight;
      ctx.beginPath();
      ctx.moveTo(plotLeft, y);
      ctx.lineTo(plotRight, y);
      ctx.stroke();
    }

    // Draw spectrum trace
    ctx.strokeStyle = COLORS.trace;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();

    const { real, ppm_axis } = spectrum;
    let firstPoint = true;

    for (let i = 0; i < real.length; i++) {
      const ppm = ppm_axis[i];
      const intensity = real[i];

      // Skip points outside view
      if (ppm < viewBounds.xMin || ppm > viewBounds.xMax) continue;

      const { x, y } = spectrumToCanvas(ppm, intensity, width, height);

      // Clamp y to plot area
      const clampedY = Math.max(plotTop, Math.min(plotBottom, y));

      if (firstPoint) {
        ctx.moveTo(x, clampedY);
        firstPoint = false;
      } else {
        ctx.lineTo(x, clampedY);
      }
    }
    ctx.stroke();

    // Draw peaks
    for (const peak of peaks) {
      if (peak.ppm < viewBounds.xMin || peak.ppm > viewBounds.xMax) continue;

      const { x, y } = spectrumToCanvas(peak.ppm, peak.intensity, width, height);
      const isSelected = peak.id === selectedPeakId;

      // Peak marker
      ctx.fillStyle = isSelected ? COLORS.peakSelected : COLORS.peak;
      ctx.beginPath();
      ctx.arc(x, Math.max(plotTop, y), isSelected ? 5 : 4, 0, Math.PI * 2);
      ctx.fill();

      // Peak label
      ctx.fillStyle = COLORS.peakLabel;
      ctx.font = '10px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText(peak.ppm.toFixed(2), x, Math.max(plotTop, y) - 8);
    }

    // Draw axes
    ctx.strokeStyle = COLORS.axis;
    ctx.lineWidth = 1;

    // Y axis
    ctx.beginPath();
    ctx.moveTo(plotLeft, plotTop);
    ctx.lineTo(plotLeft, plotBottom);
    ctx.stroke();

    // X axis
    ctx.beginPath();
    ctx.moveTo(plotLeft, plotBottom);
    ctx.lineTo(plotRight, plotBottom);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = COLORS.axis;
    ctx.font = '11px system-ui';

    // PPM labels
    ctx.textAlign = 'center';
    for (let ppm = ppmStart; ppm <= viewBounds.xMax; ppm += ppmStep) {
      const x = plotLeft + ((viewBounds.xMax - ppm) / ppmRange) * plotWidth;
      ctx.fillText(ppm.toFixed(1), x, plotBottom + 15);
    }

    // PPM axis title
    ctx.fillText('ppm', plotLeft + plotWidth / 2, height - 5);

    // Intensity labels
    ctx.textAlign = 'right';
    for (let intensity = intensityStart; intensity <= viewBounds.yMax; intensity += intensityStep) {
      const y = plotTop + ((viewBounds.yMax - intensity) / intensityRange) * plotHeight;
      ctx.fillText(formatIntensity(intensity), plotLeft - 5, y + 4);
    }
  }, [spectrum, peaks, viewBounds, selectedPeakId, lineWidth, spectrumToCanvas]);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => render();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [render]);

  // Render when data changes
  useEffect(() => {
    render();
  }, [render]);

  // Mouse handlers
  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!viewBounds) return;

      isDragging.current = true;
      dragStart.current = { x: e.clientX, y: e.clientY };
      initialBounds.current = { ...viewBounds };
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

      // Report mouse position
      const coords = canvasToSpectrum(x, y, rect.width, rect.height);
      onMouseMove?.(coords.ppm, coords.intensity);

      // Handle panning
      if (isDragging.current && initialBounds.current && onViewChange) {
        const dx = e.clientX - dragStart.current.x;
        const dy = e.clientY - dragStart.current.y;

        const ppmRange = initialBounds.current.xMax - initialBounds.current.xMin;
        const intensityRange = initialBounds.current.yMax - initialBounds.current.yMin;

        const plotWidth = rect.width - 80; // approximate
        const plotHeight = rect.height - 60;

        const ppmShift = (dx / plotWidth) * ppmRange;
        const intensityShift = -(dy / plotHeight) * intensityRange;

        onViewChange({
          xMin: initialBounds.current.xMin + ppmShift,
          xMax: initialBounds.current.xMax + ppmShift,
          yMin: initialBounds.current.yMin + intensityShift,
          yMax: initialBounds.current.yMax + intensityShift,
        });
      }
    },
    [viewBounds, canvasToSpectrum, onMouseMove, onViewChange]
  );

  const handleMouseUp = useCallback(() => {
    isDragging.current = false;
    initialBounds.current = null;
  }, []);

  const handleMouseLeave = useCallback(() => {
    isDragging.current = false;
    initialBounds.current = null;
  }, []);

  // Click to select peak
  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!onPeakSelect || !viewBounds) return;

      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      const clickCoords = canvasToSpectrum(x, y, rect.width, rect.height);

      // Find nearest peak within threshold
      const threshold = (viewBounds.xMax - viewBounds.xMin) * 0.02; // 2% of view width
      let nearestPeak: PickedPeak | null = null;
      let nearestDistance = Infinity;

      for (const peak of peaks) {
        const distance = Math.abs(peak.ppm - clickCoords.ppm);
        if (distance < threshold && distance < nearestDistance) {
          nearestDistance = distance;
          nearestPeak = peak;
        }
      }

      onPeakSelect(nearestPeak?.id ?? null);
    },
    [peaks, viewBounds, canvasToSpectrum, onPeakSelect]
  );

  // Mouse wheel zoom
  const handleWheel = useCallback(
    (e: React.WheelEvent<HTMLCanvasElement>) => {
      if (!onViewChange || !viewBounds) return;
      e.preventDefault();

      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      // Zoom center in spectrum coordinates
      const center = canvasToSpectrum(x, y, rect.width, rect.height);

      // Zoom factor
      const factor = e.deltaY > 0 ? 1.1 : 0.9;

      const newXMin = center.ppm - (center.ppm - viewBounds.xMin) * factor;
      const newXMax = center.ppm + (viewBounds.xMax - center.ppm) * factor;
      const newYMin = center.intensity - (center.intensity - viewBounds.yMin) * factor;
      const newYMax = center.intensity + (viewBounds.yMax - center.intensity) * factor;

      onViewChange({
        xMin: newXMin,
        xMax: newXMax,
        yMin: newYMin,
        yMax: newYMax,
      });
    },
    [viewBounds, canvasToSpectrum, onViewChange]
  );

  return (
    <div ref={containerRef} className="w-full h-full min-h-[300px]">
      <canvas
        ref={canvasRef}
        className="cursor-crosshair"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
        onWheel={handleWheel}
      />
    </div>
  );
}

// Helper: Calculate a "nice" step value for axis ticks
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

// Helper: Format intensity values for display
function formatIntensity(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1e6) return (value / 1e6).toFixed(1) + 'M';
  if (abs >= 1e3) return (value / 1e3).toFixed(1) + 'K';
  if (abs >= 1) return value.toFixed(0);
  return value.toExponential(1);
}
