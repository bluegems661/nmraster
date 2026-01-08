//! Data structures for NMR data representation.
//!
//! This module contains the core data types for spectra, molecules,
//! experiments, and constraints.

pub mod spectrum;
pub mod molecule;
pub mod experiment;
pub mod constraint;
pub mod spin_system;
pub mod residue_topology;

pub use spectrum::*;
pub use molecule::*;
pub use experiment::*;
pub use constraint::*;
pub use spin_system::*;
pub use residue_topology::*;
