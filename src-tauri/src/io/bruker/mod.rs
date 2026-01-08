//! Bruker TopSpin file format import.
//!
//! Supports:
//! - Parameter files: acqus, acqu2s, procs, proc2s (JCAMP-DX format)
//! - 1D processed data: pdata/1/1r (real), pdata/1/1i (imaginary)
//! - 1D FID data: fid or ser
//! - 2D processed data: pdata/1/2rr
//! - 2D FID data: ser

mod jcamp;
mod params;
mod reader_1d;
mod reader_2d;

pub use jcamp::JcampParams;
pub use params::{BrukerAcqParams, BrukerProcParams};
pub use reader_1d::import_bruker_1d;
pub use reader_2d::import_bruker_2d;

use serde::{Deserialize, Serialize};

/// Options for Bruker import.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrukerImportOptions {
    /// Path to the Bruker experiment directory (contains acqus)
    pub path: String,
    /// Import processed data (pdata) instead of FID
    pub processed: bool,
    /// Processing number (default: 1)
    pub procno: u32,
    /// Optional name override
    pub name: Option<String>,
}

impl Default for BrukerImportOptions {
    fn default() -> Self {
        Self {
            path: String::new(),
            processed: true,
            procno: 1,
            name: None,
        }
    }
}
