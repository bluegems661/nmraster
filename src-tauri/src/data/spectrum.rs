//! Spectrum data structures for 1D-4D NMR data.
//!
//! Uses ndarray for efficient multi-dimensional array operations.

use ndarray::{Array1, Array2, Array3, Array4};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Metadata common to all spectrum types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumMetadata {
    pub id: Uuid,
    pub name: String,
    pub experiment_type: ExperimentType,
    pub nucleus_types: Vec<NucleusType>,
    /// Spectral width in Hz for each dimension
    pub spectral_width_hz: Vec<f64>,
    /// Carrier frequency offset in ppm for each dimension
    pub carrier_offset_ppm: Vec<f64>,
    /// Spectrometer frequency in MHz
    pub spectrometer_frequency_mhz: f64,
    /// Number of points in each dimension (original)
    pub original_points: Vec<usize>,
    /// Number of points in each dimension (after processing)
    pub processed_points: Vec<usize>,
}

/// Supported nucleus types for NMR experiments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NucleusType {
    H1,
    C13,
    N15,
    P31,
    F19,
    Other(u8), // Isotope number for other nuclei
}

impl NucleusType {
    /// Get the gyromagnetic ratio (MHz/T)
    pub fn gyromagnetic_ratio(&self) -> f64 {
        match self {
            NucleusType::H1 => 42.576,
            NucleusType::C13 => 10.705,
            NucleusType::N15 => -4.316,
            NucleusType::P31 => 17.235,
            NucleusType::F19 => 40.052,
            NucleusType::Other(_) => 0.0, // Would need lookup table
        }
    }
}

/// Common NMR experiment types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperimentType {
    // 1D experiments
    Proton1D,
    Carbon1D,

    // 2D experiments
    HSQC,       // Generic HSQC (typically 15N-HSQC)
    HSQC_15N,   // 15N-HSQC explicitly
    HSQC_13C,   // 13C-HSQC (aliphatic)
    HMBC,
    COSY,
    TOCSY,
    NOESY,
    ROESY,

    // 3D experiments
    HNCO,
    HNCA,
    HNCACB,
    CBCACONH,
    HBHACONH,
    NOESY_HSQC,

    // HSQC-TOCSY experiments (2D and 3D)
    HSQC_TOCSY_15N,    // 2D 15N-HSQC-TOCSY (N, H_tocsy)
    HSQC_TOCSY_15N_3D, // 3D 15N-HSQC-TOCSY (N, H_backbone, H_tocsy)
    HSQC_TOCSY_13C,    // 2D 13C-HSQC-TOCSY (C, H_tocsy)
    HSQC_TOCSY_13C_3D, // 3D 13C-HSQC-TOCSY (C, H_anchor, H_tocsy)

    // 4D experiments
    HNCACO,
    CC_NOESY,

    // Relaxation experiments
    T1,
    T2,
    HetNOE,
    CPMG,
    CEST,

    // Other
    Custom(String),
}

/// 1D NMR spectrum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spectrum1D {
    pub metadata: SpectrumMetadata,
    /// Real part of the spectrum
    pub real: Array1<f64>,
    /// Imaginary part (if available)
    pub imag: Option<Array1<f64>>,
    /// PPM axis values
    pub ppm_axis: Array1<f64>,
    /// Whether the spectrum has been processed (FFT applied)
    pub is_processed: bool,
}

impl Spectrum1D {
    /// Create a new 1D spectrum from raw FID data.
    pub fn from_fid(
        fid_real: Array1<f64>,
        fid_imag: Array1<f64>,
        metadata: SpectrumMetadata,
    ) -> Self {
        let n = fid_real.len();
        let sw = metadata.spectral_width_hz[0];
        let offset = metadata.carrier_offset_ppm[0];
        let sf = metadata.spectrometer_frequency_mhz;

        // Calculate PPM axis
        let sw_ppm = sw / sf;
        let ppm_axis = Array1::linspace(
            offset + sw_ppm / 2.0,
            offset - sw_ppm / 2.0,
            n,
        );

        Self {
            metadata,
            real: fid_real,
            imag: Some(fid_imag),
            ppm_axis,
            is_processed: false,
        }
    }

    /// Get the complex data for FFT processing.
    pub fn as_complex(&self) -> Array1<Complex64> {
        match &self.imag {
            Some(imag) => {
                Array1::from_iter(
                    self.real.iter().zip(imag.iter())
                        .map(|(&r, &i)| Complex64::new(r, i))
                )
            }
            None => {
                Array1::from_iter(self.real.iter().map(|&r| Complex64::new(r, 0.0)))
            }
        }
    }

    /// Get the number of points.
    pub fn len(&self) -> usize {
        self.real.len()
    }

    /// Check if the spectrum is empty.
    pub fn is_empty(&self) -> bool {
        self.real.is_empty()
    }

    /// Find the PPM value at a given index.
    pub fn ppm_at(&self, index: usize) -> Option<f64> {
        self.ppm_axis.get(index).copied()
    }

    /// Find the index closest to a given PPM value.
    pub fn index_at_ppm(&self, ppm: f64) -> usize {
        self.ppm_axis
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((*a - ppm).abs()).partial_cmp(&((*b - ppm).abs())).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

/// 2D NMR spectrum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spectrum2D {
    pub metadata: SpectrumMetadata,
    /// 2D data array [F1, F2] - rows are F1 (indirect), columns are F2 (direct)
    pub data: Array2<f64>,
    /// PPM axis for F1 (indirect dimension)
    pub ppm_axis_f1: Array1<f64>,
    /// PPM axis for F2 (direct dimension)
    pub ppm_axis_f2: Array1<f64>,
    /// Whether the spectrum has been fully processed
    pub is_processed: bool,
}

impl Spectrum2D {
    /// Create a new 2D spectrum with proper axes.
    pub fn new(
        data: Array2<f64>,
        metadata: SpectrumMetadata,
    ) -> Self {
        let (n1, n2) = data.dim();

        let sw1 = metadata.spectral_width_hz[0];
        let sw2 = metadata.spectral_width_hz[1];
        let offset1 = metadata.carrier_offset_ppm[0];
        let offset2 = metadata.carrier_offset_ppm[1];
        let sf = metadata.spectrometer_frequency_mhz;

        let sw1_ppm = sw1 / sf;
        let sw2_ppm = sw2 / sf;

        let ppm_axis_f1 = Array1::linspace(
            offset1 + sw1_ppm / 2.0,
            offset1 - sw1_ppm / 2.0,
            n1,
        );
        let ppm_axis_f2 = Array1::linspace(
            offset2 + sw2_ppm / 2.0,
            offset2 - sw2_ppm / 2.0,
            n2,
        );

        Self {
            metadata,
            data,
            ppm_axis_f1,
            ppm_axis_f2,
            is_processed: false,
        }
    }

    /// Get dimensions (F1, F2).
    pub fn shape(&self) -> (usize, usize) {
        self.data.dim()
    }

    /// Get value at PPM coordinates.
    pub fn value_at_ppm(&self, ppm_f1: f64, ppm_f2: f64) -> f64 {
        let i1 = self.index_at_ppm_f1(ppm_f1);
        let i2 = self.index_at_ppm_f2(ppm_f2);
        self.data[[i1, i2]]
    }

    /// Find index closest to PPM in F1.
    pub fn index_at_ppm_f1(&self, ppm: f64) -> usize {
        self.ppm_axis_f1
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((*a - ppm).abs()).partial_cmp(&((*b - ppm).abs())).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Find index closest to PPM in F2.
    pub fn index_at_ppm_f2(&self, ppm: f64) -> usize {
        self.ppm_axis_f2
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((*a - ppm).abs()).partial_cmp(&((*b - ppm).abs())).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Extract a 1D slice along F2 at a given F1 index.
    pub fn slice_f2(&self, f1_index: usize) -> Array1<f64> {
        self.data.row(f1_index).to_owned()
    }

    /// Extract a 1D slice along F1 at a given F2 index.
    pub fn slice_f1(&self, f2_index: usize) -> Array1<f64> {
        self.data.column(f2_index).to_owned()
    }
}

/// 3D NMR spectrum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spectrum3D {
    pub metadata: SpectrumMetadata,
    /// 3D data array [F1, F2, F3]
    pub data: Array3<f64>,
    pub ppm_axis_f1: Array1<f64>,
    pub ppm_axis_f2: Array1<f64>,
    pub ppm_axis_f3: Array1<f64>,
    pub is_processed: bool,
}

/// 4D NMR spectrum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spectrum4D {
    pub metadata: SpectrumMetadata,
    /// 4D data array [F1, F2, F3, F4]
    pub data: Array4<f64>,
    pub ppm_axis_f1: Array1<f64>,
    pub ppm_axis_f2: Array1<f64>,
    pub ppm_axis_f3: Array1<f64>,
    pub ppm_axis_f4: Array1<f64>,
    pub is_processed: bool,
}

/// Peak detected in a spectrum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Peak {
    pub id: Uuid,
    pub spectrum_id: Uuid,
    /// Position in PPM for each dimension
    pub position_ppm: Vec<f64>,
    /// Position uncertainty in PPM
    pub position_error: Vec<f64>,
    /// Peak intensity (height)
    pub intensity: f64,
    /// Peak volume (integrated)
    pub volume: Option<f64>,
    /// Line width at half height (Hz) for each dimension
    pub line_width_hz: Vec<f64>,
    /// Signal-to-noise ratio
    pub signal_noise_ratio: f64,
    /// Whether this peak is likely an artifact
    pub is_artifact: bool,
    /// Figure of merit (quality score 0-1)
    pub figure_of_merit: f64,
    /// Optional annotation
    pub annotation: Option<String>,
}

impl Peak {
    /// Create a new peak with minimal information.
    pub fn new(spectrum_id: Uuid, position_ppm: Vec<f64>, intensity: f64) -> Self {
        let ndim = position_ppm.len();
        Self {
            id: Uuid::new_v4(),
            spectrum_id,
            position_ppm,
            position_error: vec![0.01; ndim], // Default 0.01 ppm error
            intensity,
            volume: None,
            line_width_hz: vec![10.0; ndim], // Default 10 Hz line width
            signal_noise_ratio: 0.0,
            is_artifact: false,
            figure_of_merit: 1.0,
            annotation: None,
        }
    }
}

/// Chemical shift assignment for a peak dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeakAssignment {
    pub peak_id: Uuid,
    pub dimension: usize,
    pub atom_id: Uuid,
    /// Assignment probability (for ambiguous assignments)
    pub probability: f64,
    /// Whether this is the primary assignment
    pub is_primary: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectrum_1d_creation() {
        let metadata = SpectrumMetadata {
            id: Uuid::new_v4(),
            name: "test".to_string(),
            experiment_type: ExperimentType::Proton1D,
            nucleus_types: vec![NucleusType::H1],
            spectral_width_hz: vec![5000.0],
            carrier_offset_ppm: vec![4.7],
            spectrometer_frequency_mhz: 600.0,
            original_points: vec![16384],
            processed_points: vec![16384],
        };

        let real = Array1::zeros(1024);
        let imag = Array1::zeros(1024);

        let spectrum = Spectrum1D::from_fid(real, imag, metadata);

        assert_eq!(spectrum.len(), 1024);
        assert!(!spectrum.is_empty());
        assert!(!spectrum.is_processed);
    }

    #[test]
    fn test_ppm_index_conversion() {
        let metadata = SpectrumMetadata {
            id: Uuid::new_v4(),
            name: "test".to_string(),
            experiment_type: ExperimentType::Proton1D,
            nucleus_types: vec![NucleusType::H1],
            spectral_width_hz: vec![6000.0],
            carrier_offset_ppm: vec![5.0],
            spectrometer_frequency_mhz: 600.0,
            original_points: vec![1024],
            processed_points: vec![1024],
        };

        let real = Array1::zeros(1024);
        let imag = Array1::zeros(1024);
        let spectrum = Spectrum1D::from_fid(real, imag, metadata);

        // Center should be at offset
        let center_idx = spectrum.index_at_ppm(5.0);
        let center_ppm = spectrum.ppm_at(center_idx).unwrap();
        assert!((center_ppm - 5.0).abs() < 0.1);
    }
}
