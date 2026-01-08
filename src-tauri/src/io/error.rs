//! Error types for file format I/O operations.

use thiserror::Error;

/// Errors that can occur during file format import/export.
#[derive(Error, Debug)]
pub enum IoError {
    #[error("File not found: {path}")]
    FileNotFound { path: String },

    #[error("Invalid Bruker directory structure: {reason}")]
    InvalidBrukerStructure { reason: String },

    #[error("JCAMP-DX parse error in '{param}': {reason}")]
    JcampParseError { param: String, reason: String },

    #[error("NMRPipe header error: {reason}")]
    NmrPipeHeaderError { reason: String },

    #[error("Missing required parameter: {param}")]
    MissingParameter { param: String },

    #[error("Invalid data format: {reason}")]
    InvalidDataFormat { reason: String },

    #[error("Unsupported feature: {feature}")]
    UnsupportedFeature { feature: String },

    #[error("Binary data read error: {reason}")]
    BinaryReadError { reason: String },

    #[error("Unknown file format at: {path}")]
    UnknownFormat { path: String },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for I/O operations.
pub type IoResult<T> = std::result::Result<T, IoError>;

impl serde::Serialize for IoError {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}
