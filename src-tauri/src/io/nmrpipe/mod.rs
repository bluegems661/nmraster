//! NMRPipe file format import.
//!
//! NMRPipe uses a 512-float (2048 byte) header followed by float32 data.
//! Supports both 1D (.ft1) and 2D (.ft2) processed spectra.

mod header;
mod reader_1d;
mod reader_2d;

pub use header::NmrPipeHeader;
pub use reader_1d::import_nmrpipe_1d;
pub use reader_2d::import_nmrpipe_2d;
