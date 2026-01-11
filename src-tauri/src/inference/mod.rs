//! Inference engine for simultaneous NMR analysis.
//!
//! Uses unified factor graph with belief propagation for global optimization.
//! All evidence (TOCSY, carbon typing, NOESY sequential) is processed simultaneously.

pub mod factor_graph;
pub mod belief_propagation;
pub mod scoring;
pub mod unified_assignment;

#[cfg(test)]
mod integration_test;

// Core infrastructure
pub use factor_graph::*;
pub use belief_propagation::*;

// KDE-based scoring (used by unified assignment)
pub use scoring::{ShiftScorer, KDEScorer, GaussianScorer, DynamicScorer};

// THE unified approach - all evidence in one factor graph
pub use unified_assignment::{run_unified_assignment, UnifiedAssignmentParams, UnifiedAssignmentResult, PeakType};

// Observation-based unified model (the primary API)
pub use unified_assignment::{
    Observation, ObservedDimension, GroundTruth,
    NucleusToleranceParams, ToleranceSchedule,
    ObservationAssignmentResult, run_observation_assignment,
};
