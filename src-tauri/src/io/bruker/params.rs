//! Bruker acquisition and processing parameter structures.

use std::path::Path;

use super::jcamp::JcampParams;
use crate::io::error::{IoError, IoResult};
use crate::data::spectrum::{ExperimentType, NucleusType};

/// Bruker acquisition parameters (from acqus file).
#[derive(Debug, Clone)]
pub struct BrukerAcqParams {
    /// Base frequency for dimension 1 (MHz)
    pub bf1: f64,
    /// Base frequency for dimension 2 (MHz) - for 2D
    pub bf2: Option<f64>,
    /// Spectral width in Hz
    pub sw_h: f64,
    /// Spectral width in Hz for F1 (2D)
    pub sw_h_f1: Option<f64>,
    /// Time domain points (complex pairs * 2)
    pub td: usize,
    /// TD for F1 dimension (2D)
    pub td_f1: Option<usize>,
    /// Nucleus for dimension 1 (e.g., "1H", "13C")
    pub nuc1: String,
    /// Nucleus for dimension 2
    pub nuc2: Option<String>,
    /// Pulse program name
    pub pulprog: String,
    /// Byte order: 0 = little endian, 1 = big endian
    pub bytorda: u8,
    /// Data type: 0 = int32, 2 = float64
    pub dtypa: u8,
    /// Number of dimensions
    pub parmode: usize,
    /// Experiment title
    pub title: Option<String>,
    /// Origin/offset frequency in Hz
    pub o1: f64,
    /// Origin for F1 (2D)
    pub o2: Option<f64>,
}

impl BrukerAcqParams {
    /// Load acquisition parameters from a Bruker experiment directory.
    pub fn load(exp_dir: &Path) -> IoResult<Self> {
        let acqus_path = exp_dir.join("acqus");
        let acqus = JcampParams::parse_file(&acqus_path)?;

        // Try to load 2D acquisition parameters
        let acqu2s_path = exp_dir.join("acqu2s");
        let acqu2s = JcampParams::parse_file(&acqu2s_path).ok();

        let bf1 = acqus.require_f64("BF1")?;
        let sw_h = acqus.require_f64("SW_h")?;
        let td = acqus.require_usize("TD")?;
        let bytorda = acqus.get_usize("BYTORDA").unwrap_or(0) as u8;
        let dtypa = acqus.get_usize("DTYPA").unwrap_or(0) as u8;
        let parmode = acqus.get_usize("PARMODE").unwrap_or(0);
        let o1 = acqus.get_f64("O1").unwrap_or(0.0);

        let nuc1 = acqus
            .get_str("NUC1")
            .map(|s| s.trim_matches(|c| c == '<' || c == '>').to_string())
            .unwrap_or_else(|| "1H".to_string());

        let pulprog = acqus
            .get_str("PULPROG")
            .map(|s| s.trim_matches(|c| c == '<' || c == '>').to_string())
            .unwrap_or_default();

        let title = acqus.get_str("TITLE").map(|s| s.to_string());

        // 2D parameters
        let (bf2, sw_h_f1, td_f1, nuc2, o2) = if let Some(ref acq2) = acqu2s {
            (
                acq2.get_f64("BF1"),
                acq2.get_f64("SW_h"),
                acq2.get_usize("TD"),
                acq2.get_str("NUC1").map(|s| {
                    s.trim_matches(|c| c == '<' || c == '>').to_string()
                }),
                acq2.get_f64("O1"),
            )
        } else {
            (None, None, None, None, None)
        };

        Ok(Self {
            bf1,
            bf2,
            sw_h,
            sw_h_f1,
            td,
            td_f1,
            nuc1,
            nuc2,
            pulprog,
            bytorda,
            dtypa,
            parmode,
            title,
            o1,
            o2,
        })
    }

    /// Parse nucleus string to NucleusType.
    pub fn parse_nucleus(nuc: &str) -> NucleusType {
        let nuc = nuc.trim_matches(|c| c == '<' || c == '>');
        match nuc {
            "1H" | "H1" => NucleusType::H1,
            "13C" | "C13" => NucleusType::C13,
            "15N" | "N15" => NucleusType::N15,
            "31P" | "P31" => NucleusType::P31,
            "19F" | "F19" => NucleusType::F19,
            _ => NucleusType::Other(0),
        }
    }

    /// Infer experiment type from pulse program name.
    pub fn infer_experiment_type(&self) -> ExperimentType {
        let pulprog = self.pulprog.to_lowercase();

        // 2D experiments
        if pulprog.contains("hsqc") {
            if self.nuc2.as_ref().map(|n| n.contains("15N") || n.contains("N15")).unwrap_or(false) {
                return ExperimentType::HSQC_15N;
            }
            if self.nuc2.as_ref().map(|n| n.contains("13C") || n.contains("C13")).unwrap_or(false) {
                return ExperimentType::HSQC_13C;
            }
            return ExperimentType::HSQC;
        }
        if pulprog.contains("hmbc") {
            return ExperimentType::HMBC;
        }
        if pulprog.contains("cosy") {
            return ExperimentType::COSY;
        }
        if pulprog.contains("tocsy") {
            return ExperimentType::TOCSY;
        }
        if pulprog.contains("noesy") {
            if pulprog.contains("hsqc") {
                return ExperimentType::NOESY_HSQC;
            }
            return ExperimentType::NOESY;
        }
        if pulprog.contains("roesy") {
            return ExperimentType::ROESY;
        }

        // 3D experiments
        if pulprog.contains("hnco") {
            return ExperimentType::HNCO;
        }
        if pulprog.contains("hnca") {
            if pulprog.contains("cb") || pulprog.contains("cacb") {
                return ExperimentType::HNCACB;
            }
            return ExperimentType::HNCA;
        }
        if pulprog.contains("cbcaconh") {
            return ExperimentType::CBCACONH;
        }

        // 1D experiments
        if self.parmode == 0 {
            match Self::parse_nucleus(&self.nuc1) {
                NucleusType::H1 => return ExperimentType::Proton1D,
                NucleusType::C13 => return ExperimentType::Carbon1D,
                _ => {}
            }
        }

        ExperimentType::Custom(self.pulprog.clone())
    }
}

/// Bruker processing parameters (from procs file).
#[derive(Debug, Clone)]
pub struct BrukerProcParams {
    /// Size of processed spectrum (real points)
    pub si: usize,
    /// Size for F1 dimension (2D)
    pub si_f1: Option<usize>,
    /// Spectral width in ppm (calculated from SW_p or derived)
    pub sw_p: f64,
    /// SW in ppm for F1
    pub sw_p_f1: Option<f64>,
    /// Offset in ppm (rightmost point of spectrum)
    pub offset: f64,
    /// Offset for F1
    pub offset_f1: Option<f64>,
    /// Spectrometer frequency used for processing
    pub sf: f64,
    /// SF for F1
    pub sf_f1: Option<f64>,
    /// Byte order for processed data
    pub bytordp: u8,
    /// Data type for processed data: NC_proc scaling
    pub nc_proc: i32,
}

impl BrukerProcParams {
    /// Load processing parameters from a pdata directory.
    pub fn load(pdata_dir: &Path) -> IoResult<Self> {
        let procs_path = pdata_dir.join("procs");
        let procs = JcampParams::parse_file(&procs_path)?;

        // Try to load 2D processing parameters
        let proc2s_path = pdata_dir.join("proc2s");
        let proc2s = JcampParams::parse_file(&proc2s_path).ok();

        let si = procs.require_usize("SI")?;
        let sf = procs.require_f64("SF")?;
        let offset = procs.get_f64("OFFSET").unwrap_or(0.0);
        let sw_p = procs.get_f64("SW_p").unwrap_or(10.0);
        let bytordp = procs.get_usize("BYTORDP").unwrap_or(0) as u8;
        let nc_proc = procs.get_i64("NC_proc").unwrap_or(0) as i32;

        // 2D parameters
        let (si_f1, sw_p_f1, offset_f1, sf_f1) = if let Some(ref proc2) = proc2s {
            (
                proc2.get_usize("SI"),
                proc2.get_f64("SW_p"),
                proc2.get_f64("OFFSET"),
                proc2.get_f64("SF"),
            )
        } else {
            (None, None, None, None)
        };

        Ok(Self {
            si,
            si_f1,
            sw_p,
            sw_p_f1,
            offset,
            offset_f1,
            sf,
            sf_f1,
            bytordp,
            nc_proc,
        })
    }
}
