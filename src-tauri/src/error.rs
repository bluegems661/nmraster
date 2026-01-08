//! Error types for the NMRaster application.
//!
//! Uses thiserror for ergonomic error handling with automatic From implementations.

use thiserror::Error;

/// Top-level error type for the NMRaster application.
#[derive(Error, Debug)]
pub enum NmrError {
    #[error("Spectrum error: {0}")]
    Spectrum(#[from] SpectrumError),

    #[error("Database error: {0}")]
    Database(#[from] DatabaseError),

    #[error("Inference error: {0}")]
    Inference(#[from] InferenceError),

    #[error("Model error: {0}")]
    Model(#[from] ModelError),

    #[error("File format error: {0}")]
    FileFormat(#[from] crate::io::IoError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Errors related to spectrum processing and manipulation.
#[derive(Error, Debug)]
pub enum SpectrumError {
    #[error("Invalid spectrum dimensions: expected {expected}, got {actual}")]
    InvalidDimensions { expected: usize, actual: usize },

    #[error("FFT error: {0}")]
    Fft(String),

    #[error("Phase correction failed: {0}")]
    PhaseCorrection(String),

    #[error("Baseline correction failed: {0}")]
    BaselineCorrection(String),

    #[error("Invalid data range: {0}")]
    InvalidRange(String),

    #[error("Unsupported file format: {0}")]
    UnsupportedFormat(String),

    #[error("Data parsing error: {0}")]
    ParseError(String),

    #[error("Empty spectrum data")]
    EmptyData,
}

/// Errors related to database operations.
#[derive(Error, Debug)]
pub enum DatabaseError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("Migration error: {0}")]
    Migration(String),

    #[error("Entity not found: {entity_type} with id {id}")]
    NotFound { entity_type: String, id: String },

    #[error("Duplicate entry: {0}")]
    DuplicateEntry(String),

    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),

    #[error("Connection pool error: {0}")]
    ConnectionPool(String),
}

/// Errors related to the inference engine (factor graphs, belief propagation).
#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Belief propagation failed to converge after {iterations} iterations")]
    ConvergenceFailure { iterations: usize },

    #[error("Invalid factor graph structure: {0}")]
    InvalidGraph(String),

    #[error("Missing variable: {0}")]
    MissingVariable(String),

    #[error("Invalid potential function: {0}")]
    InvalidPotential(String),

    #[error("Assignment conflict: {0}")]
    AssignmentConflict(String),

    #[error("Empty domain for variable: {0}")]
    EmptyDomain(String),
}

/// Errors related to ML model loading and inference.
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("ONNX runtime error: {0}")]
    OnnxRuntime(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Invalid model version: {0}")]
    InvalidVersion(String),

    #[error("Model hash mismatch: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },

    #[error("Invalid input shape: expected {expected:?}, got {actual:?}")]
    InvalidInputShape { expected: Vec<i64>, actual: Vec<i64> },

    #[error("Inference failed: {0}")]
    InferenceFailed(String),
}

/// Result type alias for NMRaster operations.
pub type Result<T> = std::result::Result<T, NmrError>;

// Implement conversion for Tauri commands
impl serde::Serialize for NmrError {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = NmrError::Spectrum(SpectrumError::InvalidDimensions {
            expected: 2,
            actual: 3,
        });
        assert!(err.to_string().contains("Invalid spectrum dimensions"));
    }

    #[test]
    fn test_error_from_spectrum() {
        let spectrum_err = SpectrumError::EmptyData;
        let nmr_err: NmrError = spectrum_err.into();
        assert!(matches!(nmr_err, NmrError::Spectrum(_)));
    }
}
