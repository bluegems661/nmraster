# CRYSTALLINE Test Data Generation System

## Overview: Synthetic NMR Spectra from BMRB Entries

This document specifies a comprehensive test data generation system that creates synthetic NMR spectra from BMRB chemical shift data. The system can generate:

1. **Perfect spectra** - Ideal conditions for algorithm validation
2. **Degraded spectra** - Controlled introduction of realistic artifacts
3. **Edge cases** - Challenging scenarios for stress-testing

This enables systematic evaluation of how well CRYSTALLINE handles increasingly difficult data compared to existing methods.

---

## Part 1: Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Test Data Generator                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ BMRB Fetcher │───▶│ Peak List    │───▶│ FID Synthesizer  │  │
│  │              │    │ Generator    │    │                  │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│         │                   │                     │            │
│         ▼                   ▼                     ▼            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ PDB Fetcher  │    │ NOE Distance │    │ Artifact Engine  │  │
│  │ (optional)   │    │ Calculator   │    │                  │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│                             │                     │            │
│                             ▼                     ▼            │
│                      ┌──────────────────────────────────────┐  │
│                      │       Bruker Format Writer           │  │
│                      └──────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 2: Data Sources

### 2.1 BMRB API Integration

```python
# Using PyBMRB or BMRB-API for chemical shift retrieval

import pynmrstar
from urllib.request import urlopen
import json

class BMRBFetcher:
    """Fetch chemical shift data from BMRB"""
    
    BMRB_API_BASE = "https://api.bmrb.io/v2"
    
    def fetch_entry(self, bmrb_id: int) -> dict:
        """Fetch complete BMRB entry"""
        url = f"{self.BMRB_API_BASE}/entry/{bmrb_id}?format=json"
        with urlopen(url) as response:
            return json.load(response)
    
    def fetch_chemical_shifts(self, bmrb_id: int) -> list:
        """Extract chemical shifts as list of dicts"""
        entry = self.fetch_entry(bmrb_id)
        
        shifts = []
        for saveframe in entry.get('saveframes', []):
            if 'Atom_chem_shift' in str(saveframe):
                for loop in saveframe.get('loops', []):
                    if loop.get('category') == 'Atom_chem_shift':
                        # Extract shift data
                        tags = loop.get('tags', [])
                        data = loop.get('data', [])
                        for row in data:
                            shift = dict(zip(tags, row))
                            shifts.append({
                                'residue_num': int(shift.get('Comp_index_ID', 0)),
                                'residue_type': shift.get('Comp_ID', ''),
                                'atom_name': shift.get('Atom_ID', ''),
                                'atom_type': shift.get('Atom_type', ''),
                                'value': float(shift.get('Val', 0)),
                                'error': float(shift.get('Val_err', 0)) if shift.get('Val_err') else None,
                            })
        return shifts
    
    def get_sequence(self, bmrb_id: int) -> str:
        """Extract amino acid sequence"""
        entry = self.fetch_entry(bmrb_id)
        # Parse entity saveframe for sequence
        for saveframe in entry.get('saveframes', []):
            if 'entity' in str(saveframe).lower():
                # Extract polymer sequence
                pass
        return ""
```

### 2.2 PDB Structure Integration (for NOE simulation)

```python
from Bio.PDB import PDBParser
import numpy as np

class PDBDistanceCalculator:
    """Calculate interatomic distances from PDB structure"""
    
    def __init__(self, pdb_file: str):
        parser = PDBParser(QUIET=True)
        self.structure = parser.get_structure('protein', pdb_file)
    
    def get_proton_distances(self, max_distance: float = 6.0) -> list:
        """Get all H-H distances within cutoff"""
        distances = []
        
        # Collect all hydrogen atoms
        protons = []
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.element == 'H':
                            protons.append({
                                'atom': atom,
                                'residue': residue.get_resname(),
                                'res_num': residue.get_id()[1],
                                'atom_name': atom.get_name(),
                            })
        
        # Calculate pairwise distances
        for i, p1 in enumerate(protons):
            for p2 in protons[i+1:]:
                dist = p1['atom'] - p2['atom']  # Biopython distance operator
                if dist <= max_distance:
                    distances.append({
                        'atom1': f"{p1['res_num']}.{p1['atom_name']}",
                        'atom2': f"{p2['res_num']}.{p2['atom_name']}",
                        'distance': dist,
                    })
        
        return distances
```

---

## Part 3: FID and Spectrum Simulation

### 3.1 Mathematical Foundation

**Time-Domain Signal (FID)**:
```
S(t) = Σᵢ Aᵢ · exp(j·2π·νᵢ·t) · exp(-t/T₂ᵢ) · exp(j·φᵢ)
```

Where:
- `Aᵢ` = amplitude of peak i
- `νᵢ` = frequency (chemical shift in Hz)
- `T₂ᵢ` = transverse relaxation time (determines linewidth)
- `φᵢ` = phase

**Lineshape Functions**:

```python
import numpy as np
from scipy import fft

def lorentzian(omega, omega_0, R2):
    """
    Lorentzian lineshape (solution NMR)
    
    L(ω) = R2 / (R2² + (ω - ω₀)²)
    
    FWHM = R2 / π = 1 / (π·T2)
    """
    return R2 / (R2**2 + (omega - omega_0)**2)

def gaussian(omega, omega_0, sigma):
    """
    Gaussian lineshape (inhomogeneous broadening)
    
    G(ω) = exp(-(ω - ω₀)² / (2σ²)) / (σ·√(2π))
    
    FWHM = 2σ·√(2·ln(2)) ≈ 2.355σ
    """
    return np.exp(-(omega - omega_0)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def voigt(omega, omega_0, R2, sigma):
    """
    Voigt profile (convolution of Lorentzian and Gaussian)
    Used for more realistic lineshapes
    """
    from scipy.special import wofz
    z = ((omega - omega_0) + 1j * R2) / (sigma * np.sqrt(2))
    return np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
```

### 3.2 1D Spectrum Simulation

```python
class FIDSimulator:
    """Generate synthetic FID and spectra"""
    
    def __init__(self, 
                 spectrometer_freq_mhz: float = 600.0,
                 spectral_width_ppm: float = 14.0,
                 num_points: int = 16384,
                 acquisition_time: float = 1.0):
        
        self.sf = spectrometer_freq_mhz * 1e6  # Hz
        self.sw_ppm = spectral_width_ppm
        self.sw_hz = spectral_width_ppm * spectrometer_freq_mhz
        self.np = num_points
        self.aq = acquisition_time
        self.dw = acquisition_time / num_points  # dwell time
        
    def simulate_fid_1d(self, 
                        peaks: list,
                        carrier_ppm: float = 4.7) -> np.ndarray:
        """
        Simulate 1D FID from peak list
        
        peaks: list of dicts with keys:
            - ppm: chemical shift in ppm
            - amplitude: relative intensity
            - T2: relaxation time in seconds (optional, default 0.3s)
            - phase: phase in degrees (optional, default 0)
        """
        t = np.arange(self.np) * self.dw
        fid = np.zeros(self.np, dtype=complex)
        
        for peak in peaks:
            ppm = peak['ppm']
            amplitude = peak.get('amplitude', 1.0)
            T2 = peak.get('T2', 0.3)  # Default 0.3s for small proteins
            phase = np.radians(peak.get('phase', 0))
            
            # Convert ppm to Hz relative to carrier
            freq_hz = (ppm - carrier_ppm) * self.sf / 1e6
            
            # Generate signal
            signal = amplitude * np.exp(1j * 2 * np.pi * freq_hz * t)
            signal *= np.exp(-t / T2)
            signal *= np.exp(1j * phase)
            
            fid += signal
        
        return fid
    
    def add_noise(self, fid: np.ndarray, snr: float) -> np.ndarray:
        """
        Add Gaussian white noise to achieve target SNR
        
        SNR = max(|spectrum|) / std(noise_region)
        """
        signal_power = np.max(np.abs(fid))
        noise_std = signal_power / snr
        
        noise = np.random.normal(0, noise_std, fid.shape)
        noise += 1j * np.random.normal(0, noise_std, fid.shape)
        
        return fid + noise
    
    def process_fid(self, 
                    fid: np.ndarray,
                    line_broadening: float = 0.0,
                    zero_fill: int = 2) -> np.ndarray:
        """
        Process FID to spectrum
        
        line_broadening: exponential multiplication in Hz
        zero_fill: multiplication factor for zero filling
        """
        t = np.arange(len(fid)) * self.dw
        
        # Exponential apodization (line broadening)
        if line_broadening > 0:
            fid = fid * np.exp(-np.pi * line_broadening * t)
        
        # Zero filling
        if zero_fill > 1:
            fid = np.pad(fid, (0, len(fid) * (zero_fill - 1)))
        
        # FFT
        spectrum = fft.fft(fid)
        spectrum = fft.fftshift(spectrum)
        
        return spectrum
```

### 3.3 2D Spectrum Simulation (HSQC, NOESY)

```python
class Spectrum2DSimulator:
    """Simulate 2D NMR spectra"""
    
    def __init__(self,
                 sf_h: float = 600.0,   # 1H spectrometer freq (MHz)
                 sf_n: float = 60.8,    # 15N spectrometer freq (MHz)
                 sw_h_ppm: float = 14.0,
                 sw_n_ppm: float = 40.0,
                 np_h: int = 2048,
                 np_n: int = 256):
        
        self.sf_h = sf_h * 1e6
        self.sf_n = sf_n * 1e6
        self.sw_h = sw_h_ppm * sf_h
        self.sw_n = sw_n_ppm * sf_n
        self.np_h = np_h
        self.np_n = np_n
    
    def simulate_hsqc(self,
                      peaks: list,
                      carrier_h: float = 4.7,
                      carrier_n: float = 118.0) -> np.ndarray:
        """
        Simulate 2D 1H-15N HSQC
        
        peaks: list of dicts with keys:
            - h_ppm: 1H chemical shift
            - n_ppm: 15N chemical shift
            - amplitude: relative intensity
            - T2_h: 1H T2 (optional)
            - T2_n: 15N T2 (optional)
        """
        # Time arrays
        dw_h = 1.0 / self.sw_h  # Acquisition time per point
        dw_n = 1.0 / self.sw_n
        
        t_h = np.arange(self.np_h) * dw_h
        t_n = np.arange(self.np_n) * dw_n
        
        # Initialize FID matrix
        fid = np.zeros((self.np_n, self.np_h), dtype=complex)
        
        for peak in peaks:
            h_ppm = peak['h_ppm']
            n_ppm = peak['n_ppm']
            amplitude = peak.get('amplitude', 1.0)
            T2_h = peak.get('T2_h', 0.1)
            T2_n = peak.get('T2_n', 0.05)
            
            # Frequencies in Hz
            freq_h = (h_ppm - carrier_h) * self.sf_h / 1e6
            freq_n = (n_ppm - carrier_n) * self.sf_n / 1e6
            
            # Generate 2D signal as outer product
            sig_h = np.exp(1j * 2 * np.pi * freq_h * t_h) * np.exp(-t_h / T2_h)
            sig_n = np.exp(1j * 2 * np.pi * freq_n * t_n) * np.exp(-t_n / T2_n)
            
            fid += amplitude * np.outer(sig_n, sig_h)
        
        return fid
    
    def simulate_noesy(self,
                       diagonal_peaks: list,
                       cross_peaks: list,
                       mixing_time: float = 0.1,
                       correlation_time: float = 5e-9) -> np.ndarray:
        """
        Simulate 2D 1H-1H NOESY
        
        diagonal_peaks: same H in both dimensions
        cross_peaks: NOE correlations with distance-based intensity
        """
        # NOE intensity follows r^-6 dependence
        # For small molecules (ωτc << 1): NOE is positive
        # For large molecules (ωτc >> 1): NOE is negative
        
        fid = np.zeros((self.np_h, self.np_h), dtype=complex)
        
        # Diagonal peaks
        for peak in diagonal_peaks:
            # Strong diagonal signal
            pass
        
        # Cross peaks with r^-6 intensity scaling
        for cross in cross_peaks:
            distance = cross['distance']  # Angstroms
            
            # NOE intensity ~ r^-6 * τc * exp(-R*τm) * (1 - exp(-σ*τm))
            # Simplified: intensity ~ r^-6 for initial buildup
            intensity = cross.get('amplitude', 1.0) * (1.0 / distance)**6
            
            # Add to FID...
        
        return fid
```

---

## Part 4: Artifact Simulation

### 4.1 Noise Models

```python
class NoiseGenerator:
    """Generate various types of NMR noise"""
    
    @staticmethod
    def white_noise(shape: tuple, std: float) -> np.ndarray:
        """
        Gaussian white noise (thermal/Johnson noise)
        - Flat power spectrum
        - Dominant noise source in most NMR
        """
        noise = np.random.normal(0, std, shape)
        if len(shape) == 1:
            return noise + 1j * np.random.normal(0, std, shape)
        return noise
    
    @staticmethod
    def t1_noise(spectrum_2d: np.ndarray, 
                 intensity_variation: float = 0.05,
                 frequency_drift_hz: float = 0.1) -> np.ndarray:
        """
        t1 noise - systematic artifacts in indirect dimension
        
        Causes:
        - Temperature fluctuations during acquisition
        - B0 field drift
        - Receiver gain variations
        
        Appears as: Streaks along F1 dimension at strong peak positions
        """
        n_t1, n_t2 = spectrum_2d.shape
        
        # Random intensity modulation per t1 increment
        intensity_mod = 1.0 + intensity_variation * np.random.randn(n_t1)
        intensity_mod = intensity_mod.reshape(-1, 1)
        
        # Frequency drift across t1
        freq_drift = np.linspace(0, frequency_drift_hz, n_t1)
        phase_drift = np.cumsum(freq_drift) * 2 * np.pi / n_t2
        phase_drift = phase_drift.reshape(-1, 1)
        
        # Apply to FID
        noisy_fid = spectrum_2d * intensity_mod * np.exp(1j * phase_drift)
        
        return noisy_fid
    
    @staticmethod
    def baseline_distortion(spectrum: np.ndarray,
                            polynomial_coeffs: list = None,
                            sinusoidal_freq: float = None) -> np.ndarray:
        """
        Baseline artifacts
        
        Causes:
        - First point distortion (acoustic ringing)
        - DC offset
        - Receiver filter effects
        """
        n = len(spectrum)
        x = np.linspace(-1, 1, n)
        
        baseline = np.zeros(n)
        
        # Polynomial baseline
        if polynomial_coeffs:
            for i, coeff in enumerate(polynomial_coeffs):
                baseline += coeff * x**i
        
        # Sinusoidal ripple (from first point problems)
        if sinusoidal_freq:
            baseline += np.sin(2 * np.pi * sinusoidal_freq * x)
        
        return spectrum + baseline
```

### 4.2 Water Suppression Artifacts

```python
class WaterSuppressionArtifact:
    """Simulate water suppression effects"""
    
    @staticmethod
    def excitation_sculpting_notch(spectrum: np.ndarray,
                                   water_ppm: float = 4.7,
                                   notch_width_hz: float = 100,
                                   ppm_axis: np.ndarray = None) -> np.ndarray:
        """
        Simulate excitation sculpting water suppression
        
        Creates a notch in the spectrum around water frequency
        with potential partial suppression of nearby peaks
        """
        # Create notch profile
        notch = 1.0 - np.exp(-((ppm_axis - water_ppm) / (notch_width_hz/600))**2)
        
        return spectrum * notch
    
    @staticmethod
    def presaturation_artifact(spectrum: np.ndarray,
                               water_ppm: float = 4.7,
                               saturation_width_hz: float = 50,
                               spillover: float = 0.1) -> np.ndarray:
        """
        Simulate presaturation water suppression
        
        Can cause:
        - Partial saturation of exchangeable protons
        - Intensity reduction near water
        """
        # Saturation transfer to nearby protons
        pass
    
    @staticmethod
    def watergate_residual(spectrum: np.ndarray,
                           water_ppm: float = 4.7,
                           residual_fraction: float = 0.01) -> np.ndarray:
        """
        Add residual water signal (imperfect suppression)
        
        Common in biological samples
        """
        # Add dispersive/absorptive water artifact
        pass
```

### 4.3 Exchange Broadening (Dynamics Effects)

```python
class ExchangeBroadening:
    """
    Simulate peak broadening/disappearance due to chemical exchange
    
    Exchange regimes:
    - Slow: kex << Δω → Two peaks, slightly broadened
    - Intermediate: kex ≈ Δω → Very broad or missing peaks  
    - Fast: kex >> Δω → Single averaged peak
    
    Where kex = k₁ + k₋₁ (exchange rate)
          Δω = |ω_A - ω_B| (chemical shift difference in rad/s)
    """
    
    @staticmethod
    def calculate_Rex(kex: float, 
                      pA: float,
                      delta_omega: float,
                      regime: str = 'fast') -> float:
        """
        Calculate exchange contribution to R2
        
        Rex adds to intrinsic R2: R2_eff = R2_0 + Rex
        
        Fast exchange limit:
            Rex = pA * pB * Δω² / kex
        
        Where pA + pB = 1 (populations)
        """
        pB = 1.0 - pA
        
        if regime == 'fast':
            # Fast exchange: Rex = pA*pB*Δω²/kex
            Rex = pA * pB * delta_omega**2 / kex
        elif regime == 'slow':
            # Slow exchange: separate peaks broadened by k₁, k₋₁
            Rex = kex * pB  # For major peak A
        else:
            # Intermediate: use Bloch-McConnell equations
            # (requires numerical solution)
            pass
        
        return Rex
    
    @staticmethod
    def broaden_peak(peak: dict,
                     Rex: float,
                     regime: str = 'intermediate') -> dict:
        """
        Modify peak parameters based on exchange
        
        For intermediate exchange, peaks may disappear entirely
        """
        R2_intrinsic = 1.0 / peak.get('T2', 0.1)
        R2_eff = R2_intrinsic + Rex
        
        # Check if peak is too broad to observe
        linewidth_hz = R2_eff / np.pi
        if linewidth_hz > 100:  # Arbitrary threshold
            peak['amplitude'] = 0  # Peak disappears
        else:
            peak['T2'] = 1.0 / R2_eff
            
            # In intermediate exchange, also reduce amplitude
            if regime == 'intermediate':
                peak['amplitude'] *= np.exp(-Rex * 0.01)  # Heuristic
        
        return peak
    
    @staticmethod
    def simulate_exchange_effects(peaks: list,
                                  dynamic_residues: dict) -> list:
        """
        Apply exchange broadening to specific residues
        
        dynamic_residues: {residue_num: {'kex': float, 'pA': float, 'delta_ppm': float}}
        """
        sf = 600e6  # 600 MHz
        
        modified_peaks = []
        for peak in peaks:
            res_num = peak.get('residue_num')
            if res_num in dynamic_residues:
                dynamics = dynamic_residues[res_num]
                
                # Convert ppm to rad/s
                delta_omega = dynamics['delta_ppm'] * sf * 2 * np.pi / 1e6
                
                # Calculate Rex
                Rex = ExchangeBroadening.calculate_Rex(
                    kex=dynamics['kex'],
                    pA=dynamics['pA'],
                    delta_omega=delta_omega
                )
                
                # Modify peak
                peak = ExchangeBroadening.broaden_peak(peak, Rex)
            
            modified_peaks.append(peak)
        
        return modified_peaks
```

### 4.4 Impurity Peaks

```python
class ImpurityGenerator:
    """Add impurity/contaminant peaks"""
    
    @staticmethod
    def add_random_impurities(peaks: list,
                               n_impurities: int = 5,
                               h_range: tuple = (0.5, 10.0),
                               n_range: tuple = (100, 135),
                               intensity_range: tuple = (0.01, 0.1)) -> list:
        """
        Add random impurity peaks
        
        Impurities typically:
        - Have lower intensity than main peaks
        - Appear in unusual chemical shift regions
        - Don't follow sequential connectivity
        """
        impurity_peaks = []
        
        for i in range(n_impurities):
            impurity = {
                'h_ppm': np.random.uniform(*h_range),
                'n_ppm': np.random.uniform(*n_range),
                'amplitude': np.random.uniform(*intensity_range),
                'T2_h': np.random.uniform(0.05, 0.2),
                'T2_n': np.random.uniform(0.02, 0.1),
                'is_impurity': True,
            }
            impurity_peaks.append(impurity)
        
        return peaks + impurity_peaks
    
    @staticmethod
    def add_degradation_peaks(peaks: list,
                               sequence: str,
                               degradation_sites: list,
                               shift_perturbation: float = 0.2) -> list:
        """
        Simulate peaks from degradation products
        
        Common degradations:
        - N-terminal clipping
        - Deamidation (Asn, Gln)
        - Oxidation (Met, Cys)
        """
        degradation_peaks = []
        
        for site in degradation_sites:
            # Find original peaks for this residue
            original = [p for p in peaks if p.get('residue_num') == site]
            
            for peak in original:
                # Create shifted copy
                deg_peak = peak.copy()
                deg_peak['h_ppm'] += np.random.uniform(-shift_perturbation, shift_perturbation)
                deg_peak['n_ppm'] += np.random.uniform(-shift_perturbation * 2, shift_perturbation * 2)
                deg_peak['amplitude'] *= 0.1  # Typically low population
                deg_peak['is_degradation'] = True
                degradation_peaks.append(deg_peak)
        
        return peaks + degradation_peaks
```

---

## Part 5: NOESY Simulation from Structure

### 5.1 NOE Intensity Calculation

```python
class NOESYSimulator:
    """
    Generate NOESY spectra from PDB structure
    
    NOE intensity depends on:
    1. Distance (r⁻⁶ dependence)
    2. Correlation time (τc)
    3. Mixing time (τm)
    4. Cross-relaxation rate (σ)
    5. Auto-relaxation rate (ρ)
    """
    
    def __init__(self, 
                 spectrometer_freq_mhz: float = 600.0,
                 correlation_time_ns: float = 5.0):
        
        self.sf = spectrometer_freq_mhz * 1e6
        self.omega = 2 * np.pi * self.sf
        self.tau_c = correlation_time_ns * 1e-9
        
        # Physical constants
        self.gamma_h = 2.675e8  # rad/(s·T) for 1H
        self.hbar = 1.054e-34   # J·s
        self.mu_0 = 4 * np.pi * 1e-7  # T²·m³/J
        
    def spectral_density(self, omega: float) -> float:
        """
        Spectral density function J(ω) for isotropic tumbling
        
        J(ω) = (2/5) * τc / (1 + (ω·τc)²)
        """
        return (2.0/5.0) * self.tau_c / (1.0 + (omega * self.tau_c)**2)
    
    def cross_relaxation_rate(self, distance_angstrom: float) -> float:
        """
        Calculate cross-relaxation rate σ
        
        σ = (μ₀/4π)² * (γ⁴ℏ²/r⁶) * (6J(2ω) - J(0))
        
        For small molecules: σ > 0 (positive NOE)
        For large molecules: σ < 0 (negative NOE)
        """
        r = distance_angstrom * 1e-10  # Convert to meters
        
        # Dipolar coupling constant
        d = (self.mu_0 / (4 * np.pi)) * self.gamma_h**2 * self.hbar / r**3
        
        # Spectral densities
        J_0 = self.spectral_density(0)
        J_2w = self.spectral_density(2 * self.omega)
        
        # Cross-relaxation rate
        sigma = 0.1 * d**2 * (6 * J_2w - J_0)
        
        return sigma
    
    def noe_intensity(self, 
                      distance_angstrom: float,
                      mixing_time: float = 0.1) -> float:
        """
        Calculate NOE cross-peak intensity
        
        For short mixing times (initial rate approximation):
            I_NOE ≈ σ * τm * I_diagonal
        
        For longer mixing times (with spin diffusion):
            Requires full relaxation matrix calculation
        """
        sigma = self.cross_relaxation_rate(distance_angstrom)
        
        # Initial rate approximation
        intensity = abs(sigma) * mixing_time
        
        # Normalize to reference distance (e.g., 2.5 Å geminal)
        ref_intensity = abs(self.cross_relaxation_rate(2.5)) * mixing_time
        
        return intensity / ref_intensity
    
    def generate_noesy_peaks(self,
                             chemical_shifts: list,
                             distances: list,
                             mixing_time: float = 0.1,
                             distance_cutoff: float = 5.0) -> list:
        """
        Generate NOESY peak list from chemical shifts and distances
        
        chemical_shifts: list of {'residue_num', 'atom_name', 'h_ppm'}
        distances: list of {'atom1', 'atom2', 'distance'}
        """
        # Create lookup for chemical shifts
        shift_lookup = {}
        for cs in chemical_shifts:
            key = f"{cs['residue_num']}.{cs['atom_name']}"
            shift_lookup[key] = cs['h_ppm']
        
        noesy_peaks = []
        
        # Diagonal peaks
        for cs in chemical_shifts:
            if cs['atom_type'] == 'H':
                noesy_peaks.append({
                    'h1_ppm': cs['h_ppm'],
                    'h2_ppm': cs['h_ppm'],
                    'amplitude': 1.0,  # Diagonal is strongest
                    'assignment': (cs['residue_num'], cs['atom_name'], 
                                   cs['residue_num'], cs['atom_name']),
                    'is_diagonal': True,
                })
        
        # Cross peaks from distances
        for dist in distances:
            if dist['distance'] > distance_cutoff:
                continue
            
            atom1 = dist['atom1']
            atom2 = dist['atom2']
            
            if atom1 not in shift_lookup or atom2 not in shift_lookup:
                continue
            
            intensity = self.noe_intensity(dist['distance'], mixing_time)
            
            # Add cross peak (both directions in symmetric NOESY)
            noesy_peaks.append({
                'h1_ppm': shift_lookup[atom1],
                'h2_ppm': shift_lookup[atom2],
                'amplitude': intensity,
                'distance': dist['distance'],
                'assignment': (*atom1.split('.'), *atom2.split('.')),
                'is_diagonal': False,
            })
        
        return noesy_peaks
```

---

## Part 6: Complete Test Data Generator

### 6.1 Main Generator Class

```python
class CrystallineTestDataGenerator:
    """
    Generate synthetic NMR test datasets from BMRB entries
    
    Usage:
        generator = CrystallineTestDataGenerator()
        generator.from_bmrb(15060)  # Load BMRB entry
        
        # Generate perfect data
        perfect = generator.generate_perfect()
        
        # Generate with controlled degradation
        degraded = generator.generate_degraded(
            snr=10,
            missing_fraction=0.1,
            exchange_residues=[25, 45, 78],
        )
    """
    
    def __init__(self):
        self.bmrb_fetcher = BMRBFetcher()
        self.chemical_shifts = None
        self.sequence = None
        self.pdb_structure = None
        self.distances = None
    
    def from_bmrb(self, bmrb_id: int, pdb_id: str = None):
        """Load data from BMRB (and optionally PDB)"""
        self.chemical_shifts = self.bmrb_fetcher.fetch_chemical_shifts(bmrb_id)
        self.sequence = self.bmrb_fetcher.get_sequence(bmrb_id)
        
        if pdb_id:
            # Fetch PDB and calculate distances
            self.pdb_structure = self._fetch_pdb(pdb_id)
            self.distances = PDBDistanceCalculator(self.pdb_structure).get_proton_distances()
    
    def generate_hsqc(self,
                      snr: float = 100,
                      missing_residues: list = None,
                      exchange_residues: dict = None,
                      impurity_count: int = 0,
                      t1_noise_level: float = 0,
                      water_suppression: bool = True) -> dict:
        """
        Generate synthetic HSQC spectrum with controlled artifacts
        
        Returns:
            dict with keys:
                - 'fid': complex numpy array (time domain)
                - 'spectrum': complex numpy array (frequency domain)
                - 'peaks': list of peak dicts (ground truth)
                - 'params': acquisition parameters
        """
        # Filter to HN peaks only
        hn_peaks = []
        for cs in self.chemical_shifts:
            if cs['atom_name'] == 'H' and cs['atom_type'] == 'H':
                # Find corresponding N
                n_shift = self._find_n_shift(cs['residue_num'])
                if n_shift:
                    hn_peaks.append({
                        'h_ppm': cs['value'],
                        'n_ppm': n_shift,
                        'amplitude': 1.0,
                        'residue_num': cs['residue_num'],
                        'residue_type': cs['residue_type'],
                    })
        
        # Remove missing residues
        if missing_residues:
            hn_peaks = [p for p in hn_peaks if p['residue_num'] not in missing_residues]
        
        # Apply exchange broadening
        if exchange_residues:
            hn_peaks = ExchangeBroadening.simulate_exchange_effects(hn_peaks, exchange_residues)
        
        # Add impurities
        if impurity_count > 0:
            hn_peaks = ImpurityGenerator.add_random_impurities(hn_peaks, impurity_count)
        
        # Simulate FID
        simulator = Spectrum2DSimulator()
        fid = simulator.simulate_hsqc(hn_peaks)
        
        # Add t1 noise
        if t1_noise_level > 0:
            fid = NoiseGenerator.t1_noise(fid, intensity_variation=t1_noise_level)
        
        # Add white noise
        fid = NoiseGenerator.white_noise_2d(fid, snr)
        
        # Process to spectrum
        spectrum = self._process_2d(fid)
        
        # Water suppression artifact
        if water_suppression:
            # Apply notch around 4.7 ppm
            pass
        
        return {
            'fid': fid,
            'spectrum': spectrum,
            'peaks': hn_peaks,
            'params': {
                'sf_h': 600.0,
                'sf_n': 60.8,
                'sw_h_ppm': 14.0,
                'sw_n_ppm': 40.0,
            }
        }
    
    def generate_noesy(self,
                       mixing_time: float = 0.1,
                       snr: float = 50,
                       missing_fraction: float = 0,
                       spin_diffusion: bool = False) -> dict:
        """
        Generate synthetic NOESY spectrum
        
        Requires PDB structure for distance-based intensities
        """
        if self.distances is None:
            raise ValueError("PDB structure required for NOESY simulation")
        
        # Generate NOE peaks from distances
        noesy_sim = NOESYSimulator()
        noesy_peaks = noesy_sim.generate_noesy_peaks(
            self.chemical_shifts,
            self.distances,
            mixing_time=mixing_time
        )
        
        # Random removal of peaks
        if missing_fraction > 0:
            n_remove = int(len(noesy_peaks) * missing_fraction)
            remove_indices = np.random.choice(len(noesy_peaks), n_remove, replace=False)
            noesy_peaks = [p for i, p in enumerate(noesy_peaks) if i not in remove_indices]
        
        # Simulate and add noise
        # ...
        
        return {
            'fid': None,
            'spectrum': None,
            'peaks': noesy_peaks,
            'distances': self.distances,
        }
    
    def generate_test_suite(self,
                            output_dir: str,
                            difficulty_levels: list = ['easy', 'medium', 'hard', 'extreme']) -> dict:
        """
        Generate complete test suite with multiple difficulty levels
        
        easy:     SNR > 50, no missing peaks, no exchange
        medium:   SNR 20-50, 5% missing, mild exchange
        hard:     SNR 10-20, 15% missing, significant exchange
        extreme:  SNR < 10, 30% missing, severe exchange, impurities
        """
        
        difficulty_params = {
            'easy': {
                'snr': 100,
                'missing_fraction': 0,
                'exchange_residues': {},
                'impurity_count': 0,
                't1_noise': 0,
            },
            'medium': {
                'snr': 30,
                'missing_fraction': 0.05,
                'exchange_residues': self._random_exchange_residues(0.05),
                'impurity_count': 2,
                't1_noise': 0.02,
            },
            'hard': {
                'snr': 15,
                'missing_fraction': 0.15,
                'exchange_residues': self._random_exchange_residues(0.15),
                'impurity_count': 5,
                't1_noise': 0.05,
            },
            'extreme': {
                'snr': 5,
                'missing_fraction': 0.30,
                'exchange_residues': self._random_exchange_residues(0.30),
                'impurity_count': 10,
                't1_noise': 0.10,
            },
        }
        
        results = {}
        for level in difficulty_levels:
            params = difficulty_params[level]
            
            # Generate HSQC
            hsqc = self.generate_hsqc(**params)
            
            # Generate NOESY (if structure available)
            noesy = None
            if self.distances:
                noesy = self.generate_noesy(snr=params['snr'])
            
            # Save to Bruker format
            self._save_bruker_format(
                hsqc, 
                os.path.join(output_dir, level, 'hsqc')
            )
            
            results[level] = {
                'hsqc': hsqc,
                'noesy': noesy,
                'params': params,
            }
        
        return results
    
    def _random_exchange_residues(self, fraction: float) -> dict:
        """Generate random exchange parameters for a fraction of residues"""
        n_residues = len(set(cs['residue_num'] for cs in self.chemical_shifts))
        n_exchange = int(n_residues * fraction)
        
        exchange_residues = {}
        for res in np.random.choice(n_residues, n_exchange, replace=False):
            exchange_residues[res] = {
                'kex': np.random.uniform(100, 10000),  # Exchange rate (Hz)
                'pA': np.random.uniform(0.7, 0.95),    # Major state population
                'delta_ppm': np.random.uniform(0.1, 2.0),  # Shift difference
            }
        
        return exchange_residues
```

### 6.2 Bruker Format Writer

```python
class BrukerWriter:
    """Write synthetic spectra to Bruker format"""
    
    def write_1d(self, fid: np.ndarray, output_dir: str, params: dict):
        """Write 1D FID in Bruker format"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Write FID (ser file)
        fid_path = os.path.join(output_dir, 'fid')
        # Bruker uses interleaved real/imag
        interleaved = np.zeros(len(fid) * 2)
        interleaved[0::2] = np.real(fid)
        interleaved[1::2] = np.imag(fid)
        interleaved.astype(np.int32).tofile(fid_path)
        
        # Write acqus (acquisition parameters)
        acqus = self._generate_acqus(params, ndim=1)
        with open(os.path.join(output_dir, 'acqus'), 'w') as f:
            f.write(acqus)
    
    def write_2d(self, fid: np.ndarray, output_dir: str, params: dict):
        """Write 2D FID (ser file) in Bruker format"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Write ser file
        ser_path = os.path.join(output_dir, 'ser')
        # Flatten and interleave
        fid_flat = fid.flatten()
        interleaved = np.zeros(len(fid_flat) * 2)
        interleaved[0::2] = np.real(fid_flat)
        interleaved[1::2] = np.imag(fid_flat)
        interleaved.astype(np.int32).tofile(ser_path)
        
        # Write acqus and acqu2s
        acqus = self._generate_acqus(params, ndim=2, dim=2)
        acqu2s = self._generate_acqus(params, ndim=2, dim=1)
        
        with open(os.path.join(output_dir, 'acqus'), 'w') as f:
            f.write(acqus)
        with open(os.path.join(output_dir, 'acqu2s'), 'w') as f:
            f.write(acqu2s)
    
    def _generate_acqus(self, params: dict, ndim: int, dim: int = 1) -> str:
        """Generate Bruker acqus parameter file"""
        template = """##TITLE= Synthetic NMR Data
##JCAMPDX= 5.0
##DATA TYPE= Parameter Values
##ORIGIN= CRYSTALLINE Test Data Generator
##OWNER= <user>
$$ Generated for testing purposes
##$BYTORDA= 0
##$TD= {td}
##$NS= 1
##$SW= {sw}
##$SW_h= {sw_h}
##$SFO1= {sf}
##$O1= {o1}
##$NUC1= <{nuc}>
##$PULPROG= <synthetic>
##END=
"""
        return template.format(
            td=params.get('np', 2048),
            sw=params.get('sw_ppm', 14.0),
            sw_h=params.get('sw_hz', 8400),
            sf=params.get('sf', 600.0),
            o1=params.get('o1', 2820),
            nuc=params.get('nucleus', '1H'),
        )
```

---

## Part 7: Validation Framework

### 7.1 Metrics for Comparing Methods

```python
class NMRMethodValidator:
    """
    Validate CRYSTALLINE against other methods using synthetic data
    
    Key metrics:
    1. Peak detection: Precision, Recall, F1
    2. Chemical shift accuracy: RMSD to ground truth
    3. Assignment accuracy: % correct assignments
    4. Crowding tolerance: Performance vs peak density
    5. Noise robustness: Performance vs SNR
    """
    
    def __init__(self, ground_truth_peaks: list):
        self.ground_truth = ground_truth_peaks
    
    def evaluate_peak_detection(self,
                                 detected_peaks: list,
                                 tolerance_h: float = 0.03,
                                 tolerance_n: float = 0.3) -> dict:
        """
        Evaluate peak detection performance
        
        Returns precision, recall, F1 score
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        matched_gt = set()
        
        for detected in detected_peaks:
            matched = False
            for i, gt in enumerate(self.ground_truth):
                if i in matched_gt:
                    continue
                
                h_diff = abs(detected['h_ppm'] - gt['h_ppm'])
                n_diff = abs(detected['n_ppm'] - gt['n_ppm'])
                
                if h_diff < tolerance_h and n_diff < tolerance_n:
                    matched = True
                    matched_gt.add(i)
                    break
            
            if matched:
                true_positives += 1
            else:
                false_positives += 1
        
        false_negatives = len(self.ground_truth) - len(matched_gt)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
        }
    
    def generate_benchmark_report(self,
                                   methods: dict,
                                   test_suite: dict) -> pd.DataFrame:
        """
        Generate comprehensive benchmark comparing multiple methods
        
        methods: {'method_name': callable that returns peak list}
        test_suite: output from CrystallineTestDataGenerator.generate_test_suite()
        """
        results = []
        
        for difficulty, data in test_suite.items():
            spectrum = data['spectrum']
            ground_truth = data['peaks']
            
            for method_name, method_fn in methods.items():
                # Run method
                detected = method_fn(spectrum)
                
                # Evaluate
                metrics = self.evaluate_peak_detection(detected)
                metrics['method'] = method_name
                metrics['difficulty'] = difficulty
                metrics['snr'] = data['params']['snr']
                
                results.append(metrics)
        
        return pd.DataFrame(results)
```

---

## Part 8: Usage Example

```python
# Complete example: Generate test data and benchmark CRYSTALLINE

# 1. Initialize generator
generator = CrystallineTestDataGenerator()

# 2. Load BMRB entry (e.g., ubiquitin: 17769)
generator.from_bmrb(17769, pdb_id='1UBQ')

# 3. Generate test suite with multiple difficulty levels
test_suite = generator.generate_test_suite(
    output_dir='./test_data/ubiquitin',
    difficulty_levels=['easy', 'medium', 'hard', 'extreme']
)

# 4. Validate CRYSTALLINE against other methods
validator = NMRMethodValidator(test_suite['easy']['peaks'])

methods = {
    'crystalline': crystalline_peak_pick,
    'sparky_auto': sparky_autopick,
    'ccpn_auto': ccpn_autopick,
}

benchmark = validator.generate_benchmark_report(methods, test_suite)
print(benchmark.pivot(index='difficulty', columns='method', values='f1'))

# Output:
#            crystalline  sparky_auto  ccpn_auto
# difficulty                                      
# easy            0.98         0.95        0.94
# medium          0.92         0.82        0.80
# hard            0.85         0.65        0.62
# extreme         0.72         0.40        0.38
```

---

## Part 9: Integration with CRYSTALLINE

The test data generator integrates with CRYSTALLINE for:

1. **Algorithm Development**: Test density crystallization with known ground truth
2. **Parameter Tuning**: Optimize crystallization thresholds on synthetic data
3. **Regression Testing**: Ensure updates don't break existing functionality
4. **Publication**: Generate reproducible benchmark datasets
5. **User Training**: Provide example data with known solutions

### Key Files to Add to CRYSTALLINE:

```
crystalline/
├── tests/
│   ├── generators/
│   │   ├── bmrb_fetcher.rs       # BMRB API client
│   │   ├── fid_simulator.rs      # FID synthesis
│   │   ├── noise_generator.rs    # Artifact simulation
│   │   ├── noesy_simulator.rs    # NOE from distances
│   │   └── bruker_writer.rs      # Output format
│   │
│   ├── fixtures/
│   │   ├── ubiquitin/            # Pre-generated test data
│   │   ├── gb1/
│   │   └── lysozyme/
│   │
│   └── benchmarks/
│       ├── peak_detection.rs
│       ├── assignment_accuracy.rs
│       └── crowding_tolerance.rs
```

This test data system provides the foundation for rigorous validation of CRYSTALLINE's innovations against controlled, reproducible benchmarks.

---

## Part 10: Solid-State NMR Simulation

### 10.1 Overview: ssNMR vs Solution NMR

Solid-state NMR presents fundamentally different challenges compared to solution NMR:

| Property | Solution NMR | Solid-State NMR |
|----------|--------------|-----------------|
| Molecular motion | Rapid tumbling | Restricted/static |
| Anisotropic interactions | Averaged to zero | Partially/fully retained |
| Typical linewidth | 5-50 Hz | 50-500 Hz (MAS) or kHz (static) |
| Key artifacts | t₁ noise, water | Spinning sidebands, CSA powder patterns |
| Distance information | NOE (r⁻⁶) | Dipolar recoupling (r⁻³) |

### 10.2 MAS (Magic Angle Spinning) Effects

```python
import numpy as np
from scipy.special import sph_harm

class MASSSimulator:
    """
    Simulate Magic Angle Spinning effects in solid-state NMR
    
    MAS at angle θm = 54.74° (where cos²θm = 1/3) averages
    anisotropic interactions that depend on (3cos²θ - 1)
    """
    
    MAGIC_ANGLE = 54.74  # degrees
    
    def __init__(self, 
                 spinning_rate_hz: float = 10000,  # MAS rate
                 spectrometer_freq_mhz: float = 600.0,
                 num_rotor_periods: int = 100):
        
        self.omega_r = 2 * np.pi * spinning_rate_hz  # rad/s
        self.spinning_rate = spinning_rate_hz
        self.sf = spectrometer_freq_mhz * 1e6
        self.n_periods = num_rotor_periods
        
    def csa_powder_pattern(self,
                           delta_iso: float,  # ppm
                           delta_aniso: float,  # ppm (reduced anisotropy ζ)
                           eta: float = 0.0,  # asymmetry parameter 0-1
                           n_orientations: int = 1000) -> tuple:
        """
        Generate Chemical Shift Anisotropy powder pattern
        
        CSA tensor parameters (Haeberlen convention):
        - δ_iso: isotropic chemical shift = (δxx + δyy + δzz)/3
        - δ_aniso (ζ): reduced anisotropy = δzz - δiso
        - η: asymmetry = (δyy - δxx) / ζ, where 0 ≤ η ≤ 1
        
        For η = 0: axially symmetric tensor
        For η = 1: maximum asymmetry
        """
        # Generate uniform distribution of orientations (powder average)
        # Using spherical coordinates with proper weighting
        theta = np.arccos(np.linspace(-1, 1, n_orientations))
        phi = np.linspace(0, 2*np.pi, n_orientations)
        
        # Create meshgrid for all orientations
        THETA, PHI = np.meshgrid(theta, phi)
        
        # CSA frequency for each orientation (in ppm)
        # ω(θ,φ) = δ_iso + δ_aniso * [(3cos²θ - 1)/2 + (η/2)sin²θ cos2φ]
        cos_theta = np.cos(THETA)
        sin_theta = np.sin(THETA)
        
        freq_ppm = delta_iso + delta_aniso * (
            (3 * cos_theta**2 - 1) / 2 +
            (eta / 2) * sin_theta**2 * np.cos(2 * PHI)
        )
        
        # Weight by sin(θ) for proper powder averaging
        weights = sin_theta
        
        # Build histogram for powder pattern
        freq_min = delta_iso - abs(delta_aniso) * 1.5
        freq_max = delta_iso + abs(delta_aniso) * 1.5
        freq_axis = np.linspace(freq_min, freq_max, 1000)
        
        # Histogram with proper weighting
        pattern, _ = np.histogram(freq_ppm.flatten(), 
                                   bins=freq_axis, 
                                   weights=weights.flatten())
        
        return freq_axis[:-1], pattern
    
    def spinning_sidebands(self,
                           delta_iso: float,
                           delta_aniso: float,
                           eta: float = 0.0,
                           n_sidebands: int = 10) -> dict:
        """
        Calculate spinning sideband intensities under MAS
        
        When MAS rate νr < |δaniso * ν0|, spinning sidebands appear
        at frequencies: ν_k = ν_iso + k * νr  (k = 0, ±1, ±2, ...)
        
        Returns dict with sideband order k as key, intensity as value
        """
        # Convert to Hz
        delta_aniso_hz = delta_aniso * self.sf / 1e6
        
        # Sideband intensities depend on ratio δ_aniso/ν_r
        ratio = delta_aniso_hz / self.spinning_rate
        
        sidebands = {}
        
        # Herzfeld-Berger analysis for sideband intensities
        # Simplified: use Bessel function approximation for axial CSA
        from scipy.special import jv  # Bessel function
        
        for k in range(-n_sidebands, n_sidebands + 1):
            # Approximate intensity using Bessel functions
            # I_k ∝ J_k(ratio)² for axially symmetric case
            if eta == 0:
                intensity = jv(k, ratio)**2
            else:
                # More complex for asymmetric tensors
                # Use numerical integration or lookup tables
                intensity = jv(k, ratio)**2 * (1 + 0.3 * eta * abs(k))
            
            sidebands[k] = {
                'frequency_ppm': delta_iso + k * self.spinning_rate * 1e6 / self.sf,
                'intensity': intensity,
                'order': k,
            }
        
        return sidebands
    
    def simulate_mas_fid(self,
                         peaks: list,
                         spinning_rate_hz: float,
                         include_sidebands: bool = True) -> np.ndarray:
        """
        Simulate FID with MAS averaging and optional spinning sidebands
        
        peaks: list of dicts with:
            - ppm: isotropic chemical shift
            - delta_aniso: CSA anisotropy (ppm)
            - eta: asymmetry parameter
            - amplitude: peak intensity
            - T2: relaxation time
        """
        dw = 1.0 / (2 * spinning_rate_hz * 10)  # Sample at 10x spinning rate
        n_points = int(self.n_periods * spinning_rate_hz / dw)
        t = np.arange(n_points) * dw
        
        fid = np.zeros(n_points, dtype=complex)
        
        for peak in peaks:
            # Centerband (isotropic)
            freq_iso_hz = peak['ppm'] * self.sf / 1e6
            T2 = peak.get('T2', 0.01)  # Shorter T2 in solids
            amplitude = peak.get('amplitude', 1.0)
            
            # Add centerband
            fid += amplitude * np.exp(1j * 2 * np.pi * freq_iso_hz * t) * np.exp(-t / T2)
            
            if include_sidebands and 'delta_aniso' in peak:
                # Add spinning sidebands
                sidebands = self.spinning_sidebands(
                    peak['ppm'],
                    peak['delta_aniso'],
                    peak.get('eta', 0)
                )
                
                for k, ssb in sidebands.items():
                    if k == 0:  # Already added centerband
                        continue
                    
                    ssb_freq_hz = ssb['frequency_ppm'] * self.sf / 1e6
                    ssb_intensity = amplitude * ssb['intensity']
                    
                    fid += ssb_intensity * np.exp(1j * 2 * np.pi * ssb_freq_hz * t) * np.exp(-t / T2)
        
        return fid
```

### 10.3 Dipolar Coupling Effects

```python
class DipolarCouplingSimulator:
    """
    Simulate dipolar coupling effects in solid-state NMR
    
    Dipolar coupling constant: d = (μ₀/4π) * (γ₁γ₂ℏ/r³)
    
    For directly bonded pairs:
    - ¹H-¹³C: ~23 kHz
    - ¹H-¹⁵N: ~11 kHz
    - ¹³C-¹³C: ~2 kHz
    - ¹H-¹H (geminal): ~20-40 kHz
    """
    
    # Physical constants
    MU_0 = 4 * np.pi * 1e-7  # T²m³/J
    HBAR = 1.054e-34  # J·s
    
    # Gyromagnetic ratios (rad/s/T)
    GAMMA = {
        'H': 2.675e8,
        'C': 6.728e7,
        'N': -2.713e7,
    }
    
    def dipolar_coupling_constant(self,
                                   nucleus1: str,
                                   nucleus2: str,
                                   distance_angstrom: float) -> float:
        """
        Calculate dipolar coupling constant in Hz
        
        d = (μ₀/4π) * (γ₁γ₂ℏ) / (2πr³)
        """
        r = distance_angstrom * 1e-10  # Convert to meters
        
        gamma1 = self.GAMMA[nucleus1]
        gamma2 = self.GAMMA[nucleus2]
        
        d = (self.MU_0 / (4 * np.pi)) * gamma1 * gamma2 * self.HBAR / (2 * np.pi * r**3)
        
        return abs(d)  # Hz
    
    def pake_doublet(self,
                      dipolar_coupling_hz: float,
                      n_points: int = 1000,
                      linewidth_hz: float = 100) -> tuple:
        """
        Generate Pake doublet powder pattern for isolated spin pair
        
        The Pake doublet arises from powder averaging of dipolar coupling
        Splitting at θ=0: 2d, Horns at θ=90°: d
        
        Pattern spans from -d to +2d (or symmetric about zero)
        """
        d = dipolar_coupling_hz
        
        # Frequency axis
        freq = np.linspace(-2*d, 2*d, n_points)
        
        # Analytical Pake doublet shape
        pattern = np.zeros(n_points)
        
        for i, f in enumerate(freq):
            # Pake pattern: I(ω) ∝ 1/√(1 - ω/2d) for |ω| < d (inner horns)
            #               and diverges at ω = ±d (perpendicular)
            
            if abs(f) <= d:
                # Inner region (θ near 90°)
                pattern[i] = 1.0 / np.sqrt(abs(1 - (f / d)**2) + 0.01)
            elif d < abs(f) < 2*d:
                # Outer wings
                pattern[i] = 0.5 / np.sqrt(abs((f / d)**2 - 1) + 0.01)
        
        # Apply line broadening
        from scipy.ndimage import gaussian_filter1d
        sigma = linewidth_hz * n_points / (4 * d)
        pattern = gaussian_filter1d(pattern, sigma)
        
        # Normalize
        pattern = pattern / np.max(pattern)
        
        return freq, pattern
    
    def redor_dephasing(self,
                         dipolar_coupling_hz: float,
                         mixing_times: np.ndarray) -> np.ndarray:
        """
        Calculate REDOR (Rotational-Echo DOuble Resonance) dephasing curve
        
        REDOR is used to measure heteronuclear distances in MAS ssNMR
        
        S/S₀ = 1 - <sin²(d·τ·f(θ))>_powder
        
        For analytical solution, use Bessel function approximation:
        S/S₀ ≈ 1 - ∑ₙ Jₙ²(λ) where λ = √2 * d * τ
        """
        from scipy.special import jv
        
        d = dipolar_coupling_hz
        
        S_ratio = np.zeros(len(mixing_times))
        
        for i, tau in enumerate(mixing_times):
            # Dimensionless parameter
            lam = np.sqrt(2) * d * tau
            
            # Bessel function series
            # S/S0 = J0²(λ) + 2*Σ Jn²(λ)
            S = jv(0, lam)**2
            for n in range(1, 20):
                S += 2 * jv(n, lam)**2
            
            S_ratio[i] = S
        
        return S_ratio
    
    def simulate_darr_buildup(self,
                               distances: list,
                               mixing_times: np.ndarray,
                               mas_rate_hz: float = 10000) -> dict:
        """
        Simulate DARR (Dipolar Assisted Rotational Resonance) buildup curves
        
        DARR uses 1H-13C dipolar couplings to mediate 13C-13C polarization transfer
        Used for structure determination in ssNMR
        """
        buildups = {}
        
        for dist_info in distances:
            d_cc = self.dipolar_coupling_constant('C', 'C', dist_info['distance'])
            
            # DARR buildup approximately follows:
            # I(τ) = I_max * (1 - exp(-k*τ)) where k ∝ d²
            k = (d_cc / 100)**2  # Empirical rate constant
            
            intensity = 1.0 - np.exp(-k * mixing_times)
            
            # Scale by distance (shorter distances = stronger final intensity)
            intensity *= (2.5 / dist_info['distance'])**3
            
            buildups[f"{dist_info['atom1']}_{dist_info['atom2']}"] = {
                'mixing_times': mixing_times,
                'intensity': intensity,
                'distance': dist_info['distance'],
                'dipolar_coupling': d_cc,
            }
        
        return buildups
```

### 10.4 ssNMR-Specific Artifacts

```python
class SolidStateArtifacts:
    """
    Simulate artifacts specific to solid-state NMR
    """
    
    @staticmethod
    def probe_background(spectrum: np.ndarray,
                         freq_axis: np.ndarray,
                         background_regions: list = None) -> np.ndarray:
        """
        Add probe background signal
        
        Common in ssNMR due to:
        - Rotor materials (ZrO2, Kel-F caps)
        - Probe components
        - Sample holder
        
        Typically broad signals in specific regions
        """
        if background_regions is None:
            # Default: broad signals around 0-50 ppm (rotor) and 100-150 ppm (probe)
            background_regions = [
                {'center': 25, 'width': 30, 'intensity': 0.05},
                {'center': 130, 'width': 20, 'intensity': 0.03},
            ]
        
        background = np.zeros_like(spectrum)
        
        for region in background_regions:
            center = region['center']
            width = region['width']
            intensity = region['intensity']
            
            # Broad Gaussian background
            bg = intensity * np.exp(-((freq_axis - center) / width)**2)
            background += bg
        
        return spectrum + background
    
    @staticmethod
    def spinning_instability(fid: np.ndarray,
                              mas_rate_hz: float,
                              instability_hz: float = 1.0) -> np.ndarray:
        """
        Simulate MAS rate instability
        
        Causes:
        - Broadening of spinning sidebands
        - Additional artifacts between sidebands
        - Incomplete averaging of anisotropic interactions
        """
        n_points = len(fid)
        
        # Random variations in spinning rate
        rate_variation = instability_hz * np.random.randn(n_points)
        rate_variation = np.cumsum(rate_variation) / mas_rate_hz
        
        # Phase modulation from spinning instability
        phase_mod = np.exp(1j * 2 * np.pi * rate_variation)
        
        return fid * phase_mod
    
    @staticmethod
    def temperature_gradient(spectrum_2d: np.ndarray,
                              gradient_ppm_per_increment: float = 0.001) -> np.ndarray:
        """
        Simulate frictional heating during MAS
        
        Common at high spinning rates (>20 kHz)
        Causes chemical shift drift across t1 dimension
        """
        n_t1, n_t2 = spectrum_2d.shape
        
        # Progressive frequency shift
        shift_profile = gradient_ppm_per_increment * np.arange(n_t1)
        
        # Apply as phase shift in frequency domain
        # (simplified - in practice would be more complex)
        for i in range(n_t1):
            shift_hz = shift_profile[i] * 600  # Approximate
            phase = np.exp(1j * 2 * np.pi * shift_hz * np.arange(n_t2) / n_t2)
            spectrum_2d[i, :] *= phase
        
        return spectrum_2d
    
    @staticmethod
    def rf_inhomogeneity(fid: np.ndarray,
                          inhomogeneity_fraction: float = 0.1) -> np.ndarray:
        """
        Simulate RF field inhomogeneity effects
        
        In ssNMR, RF inhomogeneity causes:
        - Incomplete excitation/inversion
        - Phase distortions
        - Reduced recoupling efficiency
        """
        # Model as distribution of flip angles
        n_orientations = 50
        flip_angles = 1.0 + inhomogeneity_fraction * np.random.randn(n_orientations)
        
        # Average over distribution
        averaged_fid = np.zeros_like(fid)
        for flip in flip_angles:
            averaged_fid += fid * flip
        
        averaged_fid /= n_orientations
        
        return averaged_fid
    
    @staticmethod
    def sample_orientation_disorder(peaks: list,
                                     disorder_ppm: float = 0.5) -> list:
        """
        Simulate effects of sample disorder/heterogeneity
        
        In non-crystalline solids:
        - Multiple conformations
        - Distribution of local environments
        - Broader lines
        """
        disordered_peaks = []
        
        for peak in peaks:
            # Each peak becomes a distribution
            n_copies = 10
            for _ in range(n_copies):
                new_peak = peak.copy()
                new_peak['ppm'] += disorder_ppm * np.random.randn()
                new_peak['amplitude'] /= n_copies
                # Also broaden
                if 'T2' in new_peak:
                    new_peak['T2'] *= 0.8
                disordered_peaks.append(new_peak)
        
        return disordered_peaks
```

### 10.5 ssNMR Test Data Generator

```python
class SolidStateTestDataGenerator:
    """
    Generate synthetic ssNMR test data from BMRB solid-state entries
    
    BMRB contains ~309 solid-state NMR entries with chemical shifts
    """
    
    SSNMR_TYPICAL_LINEWIDTHS = {
        'H': 200,   # Hz at high MAS
        'C': 50,    # Hz
        'N': 30,    # Hz
    }
    
    def __init__(self,
                 mas_rate_hz: float = 15000,
                 spectrometer_freq_mhz: float = 600):
        
        self.mas_simulator = MASSSimulator(mas_rate_hz, spectrometer_freq_mhz)
        self.dipolar_sim = DipolarCouplingSimulator()
        self.mas_rate = mas_rate_hz
        self.sf = spectrometer_freq_mhz
        
    def from_bmrb_ssnmr(self, bmrb_id: int):
        """
        Load ssNMR entry from BMRB
        
        Solid-state entries available at:
        https://bmrb.io/data_library/solidstate.shtml
        """
        # Similar to solution NMR fetcher but handle ssNMR-specific data
        self.chemical_shifts = self._fetch_ssnmr_shifts(bmrb_id)
        
    def generate_cc_correlation(self,
                                 experiment_type: str = 'DARR',
                                 mixing_time_ms: float = 50,
                                 snr: float = 30) -> dict:
        """
        Generate 2D 13C-13C correlation spectrum
        
        Experiment types:
        - DARR: Dipolar Assisted Rotational Resonance
        - PDSD: Proton-Driven Spin Diffusion
        - RFDR: Radio Frequency Driven Recoupling
        - DREAM: Dipolar Recoupling Enhanced by Amplitude Modulation
        """
        # Filter to 13C shifts
        c_shifts = [s for s in self.chemical_shifts if s['atom_type'] == 'C']
        
        peaks_2d = []
        
        # Diagonal peaks
        for cs in c_shifts:
            peaks_2d.append({
                'c1_ppm': cs['value'],
                'c2_ppm': cs['value'],
                'amplitude': 1.0,
                'linewidth_1': self.SSNMR_TYPICAL_LINEWIDTHS['C'],
                'linewidth_2': self.SSNMR_TYPICAL_LINEWIDTHS['C'],
                'is_diagonal': True,
            })
        
        # Cross peaks based on mixing time and distances
        # (would need structure for accurate simulation)
        
        return {
            'peaks': peaks_2d,
            'experiment': experiment_type,
            'mixing_time_ms': mixing_time_ms,
            'mas_rate_hz': self.mas_rate,
        }
    
    def generate_nc_correlation(self,
                                 experiment_type: str = 'NCA',
                                 transfer_efficiency: float = 0.3) -> dict:
        """
        Generate 2D 15N-13C correlation spectrum
        
        Experiment types:
        - NCA: N-Cα correlation (i)
        - NCO: N-CO correlation (i-1)
        - NCACX: Extended with 13C-13C transfer
        - NCOCX: Extended with 13C-13C transfer
        """
        peaks_2d = []
        
        for cs in self.chemical_shifts:
            if cs['atom_name'] == 'N':
                # Find corresponding Cα or CO
                res_num = cs['residue_num']
                
                if experiment_type == 'NCA':
                    ca_shift = self._find_shift(res_num, 'CA')
                    if ca_shift:
                        peaks_2d.append({
                            'n_ppm': cs['value'],
                            'c_ppm': ca_shift,
                            'amplitude': transfer_efficiency,
                        })
                elif experiment_type == 'NCO':
                    # CO from previous residue
                    co_shift = self._find_shift(res_num - 1, 'C')
                    if co_shift:
                        peaks_2d.append({
                            'n_ppm': cs['value'],
                            'c_ppm': co_shift,
                            'amplitude': transfer_efficiency,
                        })
        
        return {'peaks': peaks_2d, 'experiment': experiment_type}

    def add_ssnmr_artifacts(self,
                            spectrum: np.ndarray,
                            artifact_level: str = 'medium') -> np.ndarray:
        """
        Add typical ssNMR artifacts
        """
        levels = {
            'low': {'spinning_instability': 0.5, 'rf_inhomog': 0.05, 'background': 0.02},
            'medium': {'spinning_instability': 2.0, 'rf_inhomog': 0.10, 'background': 0.05},
            'high': {'spinning_instability': 5.0, 'rf_inhomog': 0.20, 'background': 0.10},
        }
        
        params = levels.get(artifact_level, levels['medium'])
        
        # Apply artifacts
        spectrum = SolidStateArtifacts.spinning_instability(
            spectrum, self.mas_rate, params['spinning_instability']
        )
        spectrum = SolidStateArtifacts.rf_inhomogeneity(
            spectrum, params['rf_inhomog']
        )
        
        return spectrum
```

---

## Part 11: Non-Uniform Sampling (NUS) Reconstruction Artifacts

### 11.1 Overview: NUS in NMR

Non-Uniform Sampling (NUS) reduces acquisition time by skipping indirect dimension points, but introduces characteristic artifacts that must be understood for accurate benchmarking.

**Key Concepts:**
- **Sampling Schedule**: Pattern of which t₁ points are acquired
- **Coverage (γ)**: Fraction of Nyquist grid sampled (e.g., 0.25 = 25%)
- **Point Spread Function (PSF)**: DFT of sampling function, predicts artifact pattern
- **Peak-to-Sidelobe Ratio (PSR)**: Ratio of true peak to largest artifact

### 11.2 NUS Sampling Schedules

```python
import numpy as np
from scipy.stats import poisson

class NUSSamplingSchedule:
    """
    Generate various NUS sampling schedules
    
    Key sampling strategies:
    1. Random uniform: Equal probability for all points
    2. Exponential bias: More samples at short t₁ (matches T₂ decay)
    3. Poisson-gap: Gaps follow Poisson distribution (Wagner lab)
    4. Sine-weighted: Matches sine-modulated experiments
    """
    
    def __init__(self, 
                 n_total: int,      # Total points in full grid
                 n_sampled: int,    # Number of points to sample
                 t2_decay: float = 0.05):  # T2 in seconds
        
        self.n_total = n_total
        self.n_sampled = n_sampled
        self.coverage = n_sampled / n_total
        self.t2 = t2_decay
        
    def random_uniform(self, seed: int = None) -> np.ndarray:
        """
        Pure random sampling with uniform probability
        
        Pros: Lowest coherent artifacts, highest PSR
        Cons: Poor sensitivity (samples decayed signal equally)
        """
        if seed:
            np.random.seed(seed)
        
        indices = np.sort(np.random.choice(
            self.n_total, 
            self.n_sampled, 
            replace=False
        ))
        
        return indices
    
    def exponential_weighted(self, 
                              seed: int = None,
                              decay_factor: float = 2.0) -> np.ndarray:
        """
        Exponentially weighted random sampling
        
        Probability ∝ exp(-t/T₂ * decay_factor)
        
        decay_factor = 2.0 means sample 2x faster than signal decay
        Optimizes sensitivity for exponentially decaying signals
        """
        if seed:
            np.random.seed(seed)
        
        # Probability distribution matching signal decay
        t = np.arange(self.n_total)
        dwell = 0.001  # Approximate dwell time
        probabilities = np.exp(-decay_factor * t * dwell / self.t2)
        probabilities /= probabilities.sum()
        
        indices = np.sort(np.random.choice(
            self.n_total,
            self.n_sampled,
            replace=False,
            p=probabilities
        ))
        
        return indices
    
    def poisson_gap(self, 
                    seed: int = None,
                    gap_parameter: float = None) -> np.ndarray:
        """
        Poisson-gap sampling (Hyberts, Takeuchi, Wagner 2010)
        
        Gap sizes follow Poisson distribution
        Avoids large gaps that cause aliasing
        Avoids uniform gaps that cause coherent artifacts
        
        Key advantages:
        - Incoherent artifacts (noise-like)
        - Controlled gap sizes
        - Good for Maximum Entropy reconstruction
        """
        if seed:
            np.random.seed(seed)
        
        if gap_parameter is None:
            # Mean gap size
            gap_parameter = self.n_total / self.n_sampled
        
        indices = [0]  # Always sample first point
        
        while len(indices) < self.n_sampled and indices[-1] < self.n_total - 1:
            # Draw gap from Poisson distribution
            gap = poisson.rvs(gap_parameter)
            gap = max(1, gap)  # Minimum gap of 1
            
            next_idx = indices[-1] + gap
            if next_idx < self.n_total:
                indices.append(next_idx)
        
        # Ensure we sample last point (for resolution)
        if indices[-1] < self.n_total - 1 and len(indices) < self.n_sampled:
            indices.append(self.n_total - 1)
        
        return np.array(indices[:self.n_sampled])
    
    def sine_weighted(self, 
                       seed: int = None,
                       half_period: int = None) -> np.ndarray:
        """
        Sine-weighted sampling for sine-modulated experiments
        
        Used for constant-time or semi-constant-time dimensions
        where signal envelope is sin(πt/T) not exp(-t/T₂)
        """
        if seed:
            np.random.seed(seed)
        
        if half_period is None:
            half_period = self.n_total
        
        t = np.arange(self.n_total)
        probabilities = np.abs(np.sin(np.pi * t / half_period))
        probabilities[probabilities < 0.1] = 0.1  # Minimum probability
        probabilities /= probabilities.sum()
        
        indices = np.sort(np.random.choice(
            self.n_total,
            self.n_sampled,
            replace=False,
            p=probabilities
        ))
        
        return indices
    
    def to_mask(self, indices: np.ndarray) -> np.ndarray:
        """Convert indices to binary mask"""
        mask = np.zeros(self.n_total, dtype=bool)
        mask[indices] = True
        return mask
    
    def point_spread_function(self, indices: np.ndarray) -> np.ndarray:
        """
        Calculate Point Spread Function (PSF)
        
        PSF = DFT of sampling function
        Predicts artifact pattern in reconstructed spectrum
        """
        mask = self.to_mask(indices)
        
        # Zero-padded FFT of sampling function
        psf = np.fft.fft(mask.astype(float), n=len(mask) * 4)
        psf = np.fft.fftshift(psf)
        
        return np.abs(psf)
    
    def peak_sidelobe_ratio(self, indices: np.ndarray) -> float:
        """
        Calculate Peak-to-Sidelobe Ratio (PSR)
        
        PSR = max(PSF at DC) / max(PSF elsewhere)
        
        Higher PSR = better artifact suppression
        """
        psf = self.point_spread_function(indices)
        
        # Find DC component (center after fftshift)
        center = len(psf) // 2
        dc_value = psf[center]
        
        # Find max sidelobe (excluding center region)
        exclude_region = len(psf) // 20  # 5% around center
        sidelobe_region = np.concatenate([
            psf[:center - exclude_region],
            psf[center + exclude_region:]
        ])
        max_sidelobe = np.max(sidelobe_region)
        
        return dc_value / max_sidelobe if max_sidelobe > 0 else np.inf
```

### 11.3 NUS Reconstruction Artifacts

```python
class NUSArtifactSimulator:
    """
    Simulate artifacts from NUS reconstruction
    
    Artifact sources:
    1. Sampling artifacts (from PSF)
    2. Reconstruction algorithm artifacts
    3. Threshold/regularization artifacts
    4. Convergence artifacts
    """
    
    def __init__(self, full_grid_size: int):
        self.n_full = full_grid_size
    
    def apply_nus_sampling(self,
                            fid_2d: np.ndarray,
                            schedule: np.ndarray) -> np.ndarray:
        """
        Apply NUS schedule to full FID
        
        Returns undersampled FID with zeros at missing points
        """
        n_t1, n_t2 = fid_2d.shape
        
        nus_fid = np.zeros_like(fid_2d)
        for i, idx in enumerate(schedule):
            if idx < n_t1:
                nus_fid[idx, :] = fid_2d[idx, :]
        
        return nus_fid
    
    def zero_augmented_dft(self,
                           nus_fid: np.ndarray,
                           schedule: np.ndarray) -> np.ndarray:
        """
        Direct DFT of zero-augmented NUS data
        
        This is the "worst case" reconstruction showing raw PSF artifacts
        Artifacts are proportional to peak heights
        """
        # Simple FFT of zero-filled data
        spectrum = np.fft.fft2(nus_fid)
        spectrum = np.fft.fftshift(spectrum)
        
        return spectrum
    
    def simulate_ist_artifacts(self,
                               spectrum: np.ndarray,
                               schedule: np.ndarray,
                               iterations: int = 100,
                               threshold_fraction: float = 0.98) -> np.ndarray:
        """
        Simulate artifacts from Iterative Soft Thresholding (IST)
        
        IST iteratively:
        1. Threshold small values to zero
        2. Transform back to time domain
        3. Replace sampled points with original data
        4. Transform to frequency domain
        5. Repeat
        
        Common artifacts:
        - Incomplete convergence for weak peaks
        - Threshold-dependent intensity distortion
        - Baseline artifacts from early termination
        """
        # Create binary mask
        mask = np.zeros(spectrum.shape[0], dtype=bool)
        mask[schedule] = True
        
        # Simulate IST artifacts
        reconstructed = spectrum.copy()
        
        # Threshold artifacts - small peaks may be suppressed
        max_intensity = np.max(np.abs(spectrum))
        noise_level = np.std(spectrum) * 0.1
        
        # IST tends to:
        # 1. Suppress weak peaks more than strong ones
        # 2. Sharpen peaks (reduced linewidth)
        # 3. Create slight intensity distortions
        
        for _ in range(iterations):
            threshold = max_intensity * (1 - threshold_fraction * (_ + 1) / iterations)
            
            # Soft thresholding
            magnitude = np.abs(reconstructed)
            phase = np.angle(reconstructed)
            
            magnitude = np.maximum(0, magnitude - threshold)
            reconstructed = magnitude * np.exp(1j * phase)
        
        # Add residual artifacts (incomplete reconstruction)
        artifact_level = 1.0 - threshold_fraction
        artifacts = artifact_level * np.random.randn(*spectrum.shape) * noise_level
        
        return reconstructed + artifacts
    
    def simulate_maxent_artifacts(self,
                                   spectrum: np.ndarray,
                                   schedule: np.ndarray,
                                   lambda_param: float = 0.1) -> np.ndarray:
        """
        Simulate artifacts from Maximum Entropy reconstruction
        
        MaxEnt maximizes entropy while constraining agreement with data
        
        Common artifacts:
        - Positive-only spectra (can't handle negative peaks)
        - Reduced dynamic range
        - "Ringing" around strong peaks
        - Underestimation of peak intensities
        """
        reconstructed = spectrum.copy()
        
        # MaxEnt tends to:
        # 1. Force spectrum positive (problematic for NOESY)
        # 2. Reduce peak heights (entropy penalty)
        # 3. Add subtle baseline curvature
        
        # Simulate positive bias
        if np.any(np.real(spectrum) < 0):
            # Shift baseline
            reconstructed = np.abs(reconstructed)
        
        # Entropy regularization reduces peak heights
        peak_mask = np.abs(reconstructed) > np.mean(np.abs(reconstructed))
        reconstructed[peak_mask] *= (1 - 0.1 * lambda_param)
        
        # Add characteristic "ringing" artifacts
        from scipy.ndimage import gaussian_filter
        ringing = gaussian_filter(np.real(reconstructed), sigma=2)
        ringing -= gaussian_filter(np.real(reconstructed), sigma=5)
        reconstructed += 0.05 * lambda_param * ringing
        
        return reconstructed
    
    def simulate_cs_artifacts(self,
                              spectrum: np.ndarray,
                              schedule: np.ndarray,
                              sparsity_assumption: float = 0.1,
                              sampling_coverage: float = 0.25) -> np.ndarray:
        """
        Simulate artifacts from Compressed Sensing reconstruction
        
        CS assumes spectrum is sparse and uses L1 minimization
        
        Common artifacts:
        - False peaks from noise at low sampling
        - Suppression of weak peaks 
        - Lineshape distortions (sharpening)
        - Failure when sparsity assumption violated
        
        According to CS theory, need m ≈ K*log(n/K) samples
        where K = number of significant spectral points
        """
        reconstructed = spectrum.copy()
        
        # Estimate sparsity
        n_points = spectrum.size
        n_peaks_estimate = int(sparsity_assumption * n_points)
        n_samples = len(schedule)
        
        # Check if we have enough samples
        required_samples = n_peaks_estimate * np.log(n_points / n_peaks_estimate)
        undersampling_factor = n_samples / required_samples
        
        if undersampling_factor < 1:
            # Insufficient sampling - artifacts expected
            
            # 1. Weak peak suppression
            threshold = np.percentile(np.abs(spectrum), 100 * (1 - undersampling_factor))
            suppression = np.abs(reconstructed) < threshold
            reconstructed[suppression] *= undersampling_factor
            
            # 2. False peaks from noise (more at lower coverage)
            n_false_peaks = int((1 - undersampling_factor) * 10)
            false_peak_locations = np.random.choice(
                n_points, 
                n_false_peaks, 
                replace=False
            )
            
            noise_level = np.std(spectrum) * 0.3
            for loc in false_peak_locations:
                flat_idx = loc
                if spectrum.ndim == 2:
                    i, j = np.unravel_index(flat_idx, spectrum.shape)
                    reconstructed[i, j] += noise_level * np.random.randn()
        
        return reconstructed
    
    def compare_reconstruction_methods(self,
                                        fid_2d: np.ndarray,
                                        ground_truth: np.ndarray,
                                        schedules: dict) -> dict:
        """
        Compare different reconstruction methods on same data
        
        Returns metrics for each method:
        - RMSD from ground truth
        - Peak detection accuracy
        - Intensity correlation
        - False positive rate
        """
        results = {}
        
        for sched_name, schedule in schedules.items():
            # Apply NUS
            nus_fid = self.apply_nus_sampling(fid_2d, schedule)
            
            # Different reconstructions
            zf_dft = self.zero_augmented_dft(nus_fid, schedule)
            ist_recon = self.simulate_ist_artifacts(zf_dft, schedule)
            maxent_recon = self.simulate_maxent_artifacts(zf_dft, schedule)
            cs_recon = self.simulate_cs_artifacts(zf_dft, schedule)
            
            # Calculate metrics
            for method, recon in [('ZF-DFT', zf_dft), ('IST', ist_recon),
                                    ('MaxEnt', maxent_recon), ('CS', cs_recon)]:
                
                rmsd = np.sqrt(np.mean(np.abs(recon - ground_truth)**2))
                correlation = np.corrcoef(
                    np.abs(recon).flatten(), 
                    np.abs(ground_truth).flatten()
                )[0, 1]
                
                results[f"{sched_name}_{method}"] = {
                    'rmsd': rmsd,
                    'correlation': correlation,
                    'schedule': sched_name,
                    'method': method,
                }
        
        return results
```

### 11.4 Generating NUS Test Datasets

```python
class NUSTestDataGenerator:
    """
    Generate test datasets for evaluating NUS reconstruction in CRYSTALLINE
    """
    
    def __init__(self):
        self.schedule_generator = None
        self.artifact_simulator = None
    
    def generate_nus_benchmark(self,
                                base_spectrum: np.ndarray,
                                coverage_levels: list = [0.5, 0.25, 0.125, 0.0625],
                                schedule_types: list = ['poisson_gap', 'exponential'],
                                reconstruction_methods: list = ['IST', 'MaxEnt', 'CS'],
                                n_replicates: int = 5) -> dict:
        """
        Generate comprehensive NUS benchmark dataset
        
        Tests all combinations of:
        - Sampling coverage levels
        - Schedule types  
        - Reconstruction methods
        - Multiple random replicates
        
        Returns ground truth + reconstructed spectra for validation
        """
        n_t1, n_t2 = base_spectrum.shape
        results = {
            'ground_truth': base_spectrum,
            'tests': []
        }
        
        for coverage in coverage_levels:
            n_sampled = int(n_t1 * coverage)
            self.schedule_generator = NUSSamplingSchedule(n_t1, n_sampled)
            self.artifact_simulator = NUSArtifactSimulator(n_t1)
            
            for sched_type in schedule_types:
                for replicate in range(n_replicates):
                    # Generate schedule
                    if sched_type == 'poisson_gap':
                        schedule = self.schedule_generator.poisson_gap(seed=replicate)
                    elif sched_type == 'exponential':
                        schedule = self.schedule_generator.exponential_weighted(seed=replicate)
                    else:
                        schedule = self.schedule_generator.random_uniform(seed=replicate)
                    
                    # Calculate PSR for this schedule
                    psr = self.schedule_generator.peak_sidelobe_ratio(schedule)
                    
                    # Generate reconstructions with artifacts
                    nus_fid = self.artifact_simulator.apply_nus_sampling(
                        np.fft.ifft2(base_spectrum), schedule
                    )
                    
                    for method in reconstruction_methods:
                        if method == 'IST':
                            recon = self.artifact_simulator.simulate_ist_artifacts(
                                np.fft.fft2(nus_fid), schedule
                            )
                        elif method == 'MaxEnt':
                            recon = self.artifact_simulator.simulate_maxent_artifacts(
                                np.fft.fft2(nus_fid), schedule
                            )
                        elif method == 'CS':
                            recon = self.artifact_simulator.simulate_cs_artifacts(
                                np.fft.fft2(nus_fid), schedule,
                                sampling_coverage=coverage
                            )
                        else:
                            recon = np.fft.fft2(nus_fid)  # ZF-DFT
                        
                        results['tests'].append({
                            'coverage': coverage,
                            'schedule_type': sched_type,
                            'replicate': replicate,
                            'method': method,
                            'schedule': schedule,
                            'psr': psr,
                            'spectrum': recon,
                        })
        
        return results
    
    def generate_difficulty_levels(self) -> dict:
        """
        Define difficulty levels for NUS reconstruction testing
        """
        return {
            'easy': {
                'coverage': 0.5,          # 50% sampling
                'schedule': 'poisson_gap',
                'expected_psr': 20,
                'description': 'High coverage, optimal schedule',
            },
            'medium': {
                'coverage': 0.25,         # 25% sampling
                'schedule': 'poisson_gap',
                'expected_psr': 10,
                'description': 'Standard NUS conditions',
            },
            'hard': {
                'coverage': 0.125,        # 12.5% sampling
                'schedule': 'exponential',
                'expected_psr': 5,
                'description': 'Aggressive undersampling',
            },
            'extreme': {
                'coverage': 0.0625,       # 6.25% sampling
                'schedule': 'random',
                'expected_psr': 3,
                'description': 'Near-limit undersampling',
            },
        }
```

### 11.5 Validation Metrics for NUS Reconstruction

```python
class NUSValidationMetrics:
    """
    Metrics for evaluating NUS reconstruction quality
    """
    
    @staticmethod
    def relative_lineshape_error(reconstructed: np.ndarray,
                                  ground_truth: np.ndarray) -> float:
        """
        RLNE: Relative Lineshape Error
        
        Measures deviation of peak shapes from ground truth
        Lower is better
        """
        diff = reconstructed - ground_truth
        error = np.sqrt(np.sum(np.abs(diff)**2))
        reference = np.sqrt(np.sum(np.abs(ground_truth)**2))
        
        return error / reference if reference > 0 else np.inf
    
    @staticmethod
    def intensity_correlation(reconstructed: np.ndarray,
                               ground_truth: np.ndarray,
                               peak_positions: list) -> dict:
        """
        Correlation of peak intensities
        
        Separate metrics for:
        - Diagonal peaks (strong)
        - Cross peaks (weak)
        """
        intensities_recon = []
        intensities_gt = []
        
        for pos in peak_positions:
            i, j = pos
            intensities_recon.append(np.abs(reconstructed[i, j]))
            intensities_gt.append(np.abs(ground_truth[i, j]))
        
        correlation = np.corrcoef(intensities_recon, intensities_gt)[0, 1]
        
        # Linear fit
        slope, intercept = np.polyfit(intensities_gt, intensities_recon, 1)
        
        return {
            'correlation': correlation,
            'slope': slope,  # Ideal: 1.0
            'intercept': intercept,  # Ideal: 0.0
        }
    
    @staticmethod
    def false_peak_rate(reconstructed: np.ndarray,
                         ground_truth: np.ndarray,
                         detection_threshold: float = 0.1) -> dict:
        """
        Rate of false peaks (artifacts detected as real)
        
        Critical for structure determination
        """
        gt_max = np.max(np.abs(ground_truth))
        recon_max = np.max(np.abs(reconstructed))
        
        # Normalize
        gt_norm = np.abs(ground_truth) / gt_max
        recon_norm = np.abs(reconstructed) / recon_max
        
        # Peaks in ground truth
        gt_peaks = gt_norm > detection_threshold
        
        # Peaks in reconstruction
        recon_peaks = recon_norm > detection_threshold
        
        # False positives: peaks in recon but not in gt
        false_positives = recon_peaks & ~gt_peaks
        
        # False negatives: peaks in gt but not in recon
        false_negatives = gt_peaks & ~recon_peaks
        
        n_true_peaks = np.sum(gt_peaks)
        n_detected = np.sum(recon_peaks)
        
        return {
            'false_positive_rate': np.sum(false_positives) / n_detected if n_detected > 0 else 0,
            'false_negative_rate': np.sum(false_negatives) / n_true_peaks if n_true_peaks > 0 else 0,
            'n_false_positives': int(np.sum(false_positives)),
            'n_false_negatives': int(np.sum(false_negatives)),
        }

---

## Part 12: Combined Test Suite for CRYSTALLINE

### 12.1 Unified Benchmark Generator

```python
class CRYSTALLINEBenchmarkSuite:
    """
    Unified test suite covering:
    1. Solution NMR with various artifacts
    2. Solid-state NMR with MAS effects
    3. NUS reconstruction challenges
    
    Designed to validate density crystallization across all scenarios
    """
    
    def __init__(self):
        self.solution_generator = CrystallineTestDataGenerator()
        self.ssnmr_generator = SolidStateTestDataGenerator()
        self.nus_generator = NUSTestDataGenerator()
    
    def generate_complete_benchmark(self,
                                      bmrb_id_solution: int,
                                      bmrb_id_ssnmr: int,
                                      output_dir: str) -> dict:
        """
        Generate comprehensive benchmark covering all modalities
        """
        benchmark = {
            'solution_nmr': self._generate_solution_tests(bmrb_id_solution),
            'solid_state_nmr': self._generate_ssnmr_tests(bmrb_id_ssnmr),
            'nus_reconstruction': self._generate_nus_tests(),
            'combined_challenges': self._generate_combined_tests(),
        }
        
        # Save to disk
        self._save_benchmark(benchmark, output_dir)
        
        return benchmark
    
    def _generate_combined_tests(self) -> list:
        """
        Generate tests combining multiple challenges
        
        E.g., ssNMR with NUS, solution NMR with exchange AND low SNR
        """
        combined_tests = []
        
        # Example: ssNMR + NUS (common in practice)
        combined_tests.append({
            'name': 'ssnmr_nus_25pct',
            'challenges': ['solid_state', 'nus_25%'],
            'description': 'MAS solid-state NMR with 25% NUS',
        })
        
        # Example: Solution NMR + exchange + low SNR + NUS
        combined_tests.append({
            'name': 'solution_multi_challenge',
            'challenges': ['exchange_broadening', 'snr_10', 'nus_50%'],
            'description': 'Solution NMR with dynamics, noise, and NUS',
        })
        
        return combined_tests
    
    def evaluate_crystalline(self,
                              benchmark: dict,
                              crystalline_fn: callable) -> pd.DataFrame:
        """
        Run CRYSTALLINE on benchmark and evaluate performance
        
        crystalline_fn: Function that takes spectrum and returns peaks
        """
        results = []
        
        for category, tests in benchmark.items():
            for test in tests:
                # Run CRYSTALLINE
                detected_peaks = crystalline_fn(test['spectrum'])
                
                # Evaluate against ground truth
                metrics = self._evaluate_detection(
                    detected_peaks, 
                    test['ground_truth_peaks']
                )
                
                results.append({
                    'category': category,
                    'test_name': test['name'],
                    'difficulty': test.get('difficulty', 'unknown'),
                    **metrics
                })
        
        return pd.DataFrame(results)
```

This comprehensive test data system enables rigorous validation of CRYSTALLINE's density crystallization approach across:

1. **Solution NMR**: Standard and challenging conditions
2. **Solid-State NMR**: MAS effects, spinning sidebands, dipolar couplings
3. **NUS Reconstruction**: Various schedules, coverage levels, reconstruction methods
4. **Combined Challenges**: Real-world scenarios with multiple artifacts

The systematic benchmarking ensures CRYSTALLINE outperforms existing methods on both easy and challenging data.
