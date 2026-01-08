//! NMRPipe header parser.
//!
//! NMRPipe files have a 512 x 4-byte float header (2048 bytes total).
//! Key parameters are at specific indices in this array.

use std::fs::File;
use std::io::{Read, BufReader, Seek, SeekFrom};
use std::path::Path;

use byteorder::{BigEndian, LittleEndian, ReadBytesExt, ByteOrder};

use crate::data::spectrum::{ExperimentType, NucleusType};
use crate::io::error::{IoError, IoResult};

/// Header size in bytes (512 floats * 4 bytes).
pub const HEADER_SIZE: usize = 2048;

/// Header indices for various parameters.
/// These are 0-indexed positions in the 512-float array.
mod idx {
    // Magic number / format identifier
    pub const FDMAGIC: usize = 0;

    // Dimension order and sizes
    pub const FDDIMCOUNT: usize = 9;
    pub const FDDIMORDER: usize = 24; // Array of 4 values

    // Direct dimension (usually F2)
    pub const FDSIZE: usize = 99;
    pub const FDSW: usize = 100;      // Spectral width (Hz)
    pub const FDORIG: usize = 101;    // Origin (Hz from carrier)
    pub const FDOBS: usize = 119;     // Observe frequency (MHz)
    pub const FDFTFLAG: usize = 220;  // FT status (0=time, 1=freq)

    // Indirect dimension 1 (usually F1)
    pub const FDSPECNUM: usize = 219; // Number of spectra (F1 size)
    pub const FDSW1: usize = 229;     // SW for F1
    pub const FDORIG1: usize = 249;   // Origin for F1
    pub const FDOBS1: usize = 218;    // Observe freq for F1
    pub const FDFTFLAG1: usize = 221; // FT status for F1

    // Additional dimensions (F3, F4)
    pub const FDSW2: usize = 11;
    pub const FDSW3: usize = 29;

    // Data parameters
    pub const FDQUADFLAG: usize = 106; // Quadrature detection
    pub const FD2DPHASE: usize = 256;  // 2D phase mode

    // Labels (character data stored as floats)
    pub const FDSRCNAME: usize = 286;  // Source file name (16 chars)
    pub const FDUSERNAME: usize = 290; // User name (16 chars)
    pub const FDTITLE: usize = 297;    // Title (60 chars)

    // Nucleus labels (stored as float-encoded chars)
    pub const FDLABEL: usize = 16;     // F2 label (8 chars)
    pub const FDLABEL1: usize = 18;    // F1 label (8 chars)
}

/// NMRPipe file header.
#[derive(Debug, Clone)]
pub struct NmrPipeHeader {
    /// Number of dimensions
    pub ndim: usize,
    /// Size of each dimension (points)
    pub size: [usize; 4],
    /// Spectral width in Hz for each dimension
    pub sw: [f64; 4],
    /// Observe frequency in MHz for each dimension
    pub obs: [f64; 4],
    /// Origin (Hz from carrier) for each dimension
    pub orig: [f64; 4],
    /// Whether FT has been applied for each dimension
    pub ftflag: [bool; 4],
    /// Nucleus labels
    pub label: [String; 4],
    /// Title
    pub title: String,
    /// Byte order (true = big endian)
    pub is_big_endian: bool,
    /// Raw header data for advanced access
    pub raw: [f32; 512],
}

impl NmrPipeHeader {
    /// Parse header from a file.
    pub fn from_file(path: &Path) -> IoResult<Self> {
        let file = File::open(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                IoError::FileNotFound {
                    path: path.display().to_string(),
                }
            } else {
                IoError::Io(e)
            }
        })?;

        let mut reader = BufReader::new(file);
        Self::parse(&mut reader)
    }

    /// Parse header from a reader.
    pub fn parse<R: Read>(reader: &mut R) -> IoResult<Self> {
        let mut raw = [0f32; 512];
        let mut bytes = [0u8; HEADER_SIZE];

        reader.read_exact(&mut bytes).map_err(|e| IoError::NmrPipeHeaderError {
            reason: format!("Failed to read header: {}", e),
        })?;

        // Detect endianness from magic number (should be 0.0 or around 2.345)
        let magic_le = LittleEndian::read_f32(&bytes[0..4]);
        let magic_be = BigEndian::read_f32(&bytes[0..4]);

        let is_big_endian = (magic_be - 0.0).abs() < 0.001 || (magic_be - 2.345).abs() < 0.01;

        // Read all floats with correct endianness
        for i in 0..512 {
            let start = i * 4;
            raw[i] = if is_big_endian {
                BigEndian::read_f32(&bytes[start..start + 4])
            } else {
                LittleEndian::read_f32(&bytes[start..start + 4])
            };
        }

        // Extract parameters
        let ndim = raw[idx::FDDIMCOUNT] as usize;
        if ndim == 0 || ndim > 4 {
            return Err(IoError::NmrPipeHeaderError {
                reason: format!("Invalid dimension count: {}", ndim),
            });
        }

        // Direct dimension size (F2)
        let size_f2 = raw[idx::FDSIZE] as usize;
        // Indirect dimension size (F1) - from SPECNUM
        let size_f1 = if ndim >= 2 {
            raw[idx::FDSPECNUM] as usize
        } else {
            1
        };

        let size = [size_f2, size_f1, 1, 1];

        // Spectral widths
        let sw = [
            raw[idx::FDSW] as f64,
            raw[idx::FDSW1] as f64,
            raw[idx::FDSW2] as f64,
            raw[idx::FDSW3] as f64,
        ];

        // Observe frequencies
        let obs = [
            raw[idx::FDOBS] as f64,
            raw[idx::FDOBS1] as f64,
            0.0,
            0.0,
        ];

        // Origins (Hz)
        let orig = [
            raw[idx::FDORIG] as f64,
            raw[idx::FDORIG1] as f64,
            0.0,
            0.0,
        ];

        // FT flags
        let ftflag = [
            raw[idx::FDFTFLAG] != 0.0,
            raw[idx::FDFTFLAG1] != 0.0,
            false,
            false,
        ];

        // Parse labels (float-encoded characters)
        let label = [
            Self::parse_label(&raw, idx::FDLABEL),
            Self::parse_label(&raw, idx::FDLABEL1),
            String::new(),
            String::new(),
        ];

        // Parse title
        let title = Self::parse_string(&raw, idx::FDTITLE, 60);

        Ok(Self {
            ndim,
            size,
            sw,
            obs,
            orig,
            ftflag,
            label,
            title,
            is_big_endian,
            raw,
        })
    }

    /// Parse a label (8 characters stored as 2 floats).
    fn parse_label(raw: &[f32; 512], start: usize) -> String {
        let mut chars = Vec::with_capacity(8);
        for i in 0..2 {
            let val = raw[start + i];
            let bytes = val.to_ne_bytes();
            for &b in &bytes {
                if b > 0 && b < 128 {
                    chars.push(b as char);
                }
            }
        }
        chars.into_iter().collect::<String>().trim().to_string()
    }

    /// Parse a string (characters stored as floats).
    fn parse_string(raw: &[f32; 512], start: usize, max_len: usize) -> String {
        let num_floats = (max_len + 3) / 4;
        let mut chars = Vec::with_capacity(max_len);

        for i in 0..num_floats {
            if start + i >= 512 {
                break;
            }
            let val = raw[start + i];
            let bytes = val.to_ne_bytes();
            for &b in &bytes {
                if b > 0 && b < 128 {
                    chars.push(b as char);
                }
            }
        }

        chars.into_iter().collect::<String>().trim().to_string()
    }

    /// Infer nucleus type from label.
    pub fn infer_nucleus(&self, dim: usize) -> NucleusType {
        let label = &self.label[dim];
        match label.to_uppercase().as_str() {
            "1H" | "H" | "HN" | "HA" => NucleusType::H1,
            "13C" | "C" | "CA" | "CB" | "CO" => NucleusType::C13,
            "15N" | "N" => NucleusType::N15,
            "31P" | "P" => NucleusType::P31,
            "19F" | "F" => NucleusType::F19,
            _ => NucleusType::H1, // Default to proton
        }
    }

    /// Infer experiment type from labels and dimensions.
    pub fn infer_experiment_type(&self) -> ExperimentType {
        if self.ndim == 1 {
            match self.infer_nucleus(0) {
                NucleusType::H1 => return ExperimentType::Proton1D,
                NucleusType::C13 => return ExperimentType::Carbon1D,
                _ => {}
            }
        }

        if self.ndim == 2 {
            let nuc0 = self.infer_nucleus(0);
            let nuc1 = self.infer_nucleus(1);

            // HSQC-type experiments
            if matches!(nuc0, NucleusType::H1) && matches!(nuc1, NucleusType::N15) {
                return ExperimentType::HSQC_15N;
            }
            if matches!(nuc0, NucleusType::H1) && matches!(nuc1, NucleusType::C13) {
                return ExperimentType::HSQC_13C;
            }
            if matches!(nuc0, NucleusType::H1) && matches!(nuc1, NucleusType::H1) {
                // Could be TOCSY, NOESY, COSY - check title
                let title_lower = self.title.to_lowercase();
                if title_lower.contains("tocsy") {
                    return ExperimentType::TOCSY;
                }
                if title_lower.contains("noesy") {
                    return ExperimentType::NOESY;
                }
                if title_lower.contains("cosy") {
                    return ExperimentType::COSY;
                }
            }
        }

        ExperimentType::Custom(self.title.clone())
    }

    /// Convert origin from Hz to ppm.
    pub fn orig_ppm(&self, dim: usize) -> f64 {
        if self.obs[dim] > 0.0 {
            self.orig[dim] / self.obs[dim]
        } else {
            0.0
        }
    }

    /// Get spectral width in ppm.
    pub fn sw_ppm(&self, dim: usize) -> f64 {
        if self.obs[dim] > 0.0 {
            self.sw[dim] / self.obs[dim]
        } else {
            self.sw[dim]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_nucleus() {
        let mut header = NmrPipeHeader {
            ndim: 2,
            size: [1024, 256, 1, 1],
            sw: [6000.0, 2000.0, 0.0, 0.0],
            obs: [600.0, 60.0, 0.0, 0.0],
            orig: [0.0, 0.0, 0.0, 0.0],
            ftflag: [true, true, false, false],
            label: ["1H".to_string(), "15N".to_string(), String::new(), String::new()],
            title: "HSQC".to_string(),
            is_big_endian: false,
            raw: [0.0; 512],
        };

        assert!(matches!(header.infer_nucleus(0), NucleusType::H1));
        assert!(matches!(header.infer_nucleus(1), NucleusType::N15));
        assert!(matches!(header.infer_experiment_type(), ExperimentType::HSQC_15N));
    }
}
