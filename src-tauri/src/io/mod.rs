//! File format I/O for importing NMR spectra.
//!
//! Supports:
//! - Bruker TopSpin (acqus/procs parameters, 1r/2rr binary data)
//! - NMRPipe (.ft1, .ft2 processed spectra)
//!
//! All formats are converted to NMRaster's internal `Spectrum1D`/`Spectrum2D` types.

pub mod error;
pub mod bruker;
pub mod nmrpipe;

pub use error::{IoError, IoResult};
pub use bruker::{import_bruker_1d, import_bruker_2d, BrukerImportOptions};
pub use nmrpipe::{import_nmrpipe_1d, import_nmrpipe_2d};
