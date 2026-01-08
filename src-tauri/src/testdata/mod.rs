//! Test data generation module using BMRB chemical shift statistics.
//!
//! This module provides tools to generate realistic synthetic NMR spectra
//! with known ground truth peaks for validation of peak picking and
//! assignment algorithms.

pub mod bmrb;
pub mod bmrb_client;
pub mod kde;
pub mod peaks;
pub mod generator;

pub use bmrb::{AtomShiftStatistics, BMRBDatabase};
pub use bmrb_client::{
    BmrbError, DepositedShift, fetch_entry_shifts, extract_sequence,
    filter_residue_range, renumber_shifts, three_to_one, one_to_three,
};
pub use kde::{KDEGrid, KDEDatabase};
pub use peaks::{GroundTruthPeak, GroundTruthPeaks, AtomAssignment};
pub use generator::{SpectrumGenerator, SpectrumGeneratorParams};
