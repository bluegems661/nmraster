#!/usr/bin/env python3
"""Generate precomputed KDE grids from BMRB shift distributions.

This script loads the raw observation data from bmrb_shift_distributions.pkl
and generates a compact binary file with precomputed KDE density values
on a 512-point grid for each residue/atom combination.

Output format (JSON for debugging, will be read by Rust):
{
    "version": "1.0",
    "n_grid_points": 512,
    "grids": {
        "ALA_CA": {
            "residue": "ALA",
            "atom": "CA",
            "grid_min": 47.0,
            "grid_max": 59.0,
            "step": 0.0234...,
            "n_observations": 93327,
            "bandwidth": 0.5,
            "values": [0.001, 0.002, ...]  // 512 density values
        },
        ...
    }
}
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde


def load_distributions(pkl_path: str) -> dict:
    """Load the raw shift distributions from pickle file."""
    import pickle
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def compute_kde_grid(
    observations: np.ndarray,
    n_points: int = 512,
    percentile_range: tuple = (0.5, 99.5),
) -> dict:
    """Compute KDE on a fixed grid.

    Args:
        observations: Array of chemical shift values
        n_points: Number of grid points
        percentile_range: Percentile range for grid bounds (filters outliers)

    Returns:
        Dictionary with grid_min, grid_max, step, bandwidth, values
    """
    # Filter outliers using percentile range
    low = np.percentile(observations, percentile_range[0])
    high = np.percentile(observations, percentile_range[1])

    # Add 10% padding to the range
    padding = (high - low) * 0.1
    grid_min = low - padding
    grid_max = high + padding

    # Create grid
    grid = np.linspace(grid_min, grid_max, n_points)
    step = (grid_max - grid_min) / (n_points - 1)

    # Compute KDE
    kde = gaussian_kde(observations)
    bandwidth = float(kde.factor * np.std(observations))

    # Evaluate on grid
    values = kde(grid)

    return {
        "grid_min": float(grid_min),
        "grid_max": float(grid_max),
        "step": float(step),
        "n_observations": len(observations),
        "bandwidth": bandwidth,
        "values": values.tolist(),
    }


def main():
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    pkl_path = project_root / "bmrb_shift_distributions.pkl"
    output_json = project_root / "src-tauri" / "data" / "bmrb_kde_grids.json"

    print(f"Loading distributions from {pkl_path}...")
    distributions = load_distributions(pkl_path)

    print(f"Found {len(distributions)} residue/atom combinations")

    # Generate grids
    grids = {}
    n_points = 512

    for (residue, atom), observations in sorted(distributions.items()):
        key = f"{residue}_{atom}"
        print(f"  Processing {key}: {len(observations)} observations...", end=" ")

        try:
            grid_data = compute_kde_grid(observations, n_points=n_points)
            grid_data["residue"] = residue
            grid_data["atom"] = atom
            grids[key] = grid_data
            print(f"range [{grid_data['grid_min']:.1f}, {grid_data['grid_max']:.1f}]")
        except Exception as e:
            print(f"FAILED: {e}")

    # Create output
    output = {
        "version": "1.0",
        "n_grid_points": n_points,
        "n_entries": len(grids),
        "grids": grids,
    }

    # Write JSON
    print(f"\nWriting {len(grids)} grids to {output_json}...")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, 'w') as f:
        json.dump(output, f, indent=None)  # No indent for smaller file

    # Report size
    size_mb = output_json.stat().st_size / (1024 * 1024)
    print(f"Output file size: {size_mb:.2f} MB")

    # Verify by loading
    print("\nVerifying output...")
    with open(output_json, 'r') as f:
        loaded = json.load(f)

    # Test a few lookups
    test_cases = [
        ("ALA", "CA", 53.0),
        ("GLY", "CA", 45.0),
        ("THR", "CB", 69.0),
    ]

    print("\nTest lookups:")
    for residue, atom, shift in test_cases:
        key = f"{residue}_{atom}"
        if key in loaded["grids"]:
            grid = loaded["grids"][key]
            # Linear interpolation
            idx = (shift - grid["grid_min"]) / grid["step"]
            idx_floor = int(idx)
            if 0 <= idx_floor < len(grid["values"]) - 1:
                frac = idx - idx_floor
                density = grid["values"][idx_floor] * (1 - frac) + grid["values"][idx_floor + 1] * frac
                print(f"  {residue} {atom} at {shift:.1f} ppm: density = {density:.4f}")
            else:
                print(f"  {residue} {atom} at {shift:.1f} ppm: out of range")
        else:
            print(f"  {residue} {atom}: not found")

    print("\nDone!")


if __name__ == "__main__":
    main()
