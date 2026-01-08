//! Model registry for managing ONNX models.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::error::{ModelError, Result};

/// Registry for managing loaded ONNX models.
pub struct ModelRegistry {
    models: RwLock<HashMap<String, Arc<LoadedModel>>>,
    model_dir: PathBuf,
}

/// A loaded ONNX model (placeholder until ort integration).
pub struct LoadedModel {
    pub model_id: String,
    pub version: String,
    pub model_hash: String,
    pub input_shape: Vec<i64>,
    pub output_shape: Vec<i64>,
}

impl ModelRegistry {
    /// Create a new model registry.
    pub fn new(model_dir: PathBuf) -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            model_dir,
        }
    }

    /// Get or load a model.
    pub fn get_or_load(&self, model_id: &str, version: &str) -> Result<Arc<LoadedModel>> {
        let key = format!("{}:{}", model_id, version);

        // Check if already loaded
        if let Some(model) = self.models.read().get(&key) {
            return Ok(Arc::clone(model));
        }

        // Load model
        let model_path = self.model_dir
            .join(model_id)
            .join(version)
            .join("model.onnx");

        if !model_path.exists() {
            return Err(ModelError::ModelNotFound(model_path.display().to_string()).into());
        }

        // Placeholder: In real implementation, load via ort
        let model = Arc::new(LoadedModel {
            model_id: model_id.to_string(),
            version: version.to_string(),
            model_hash: "placeholder".to_string(),
            input_shape: vec![-1, 64],
            output_shape: vec![-1, 10],
        });

        self.models.write().insert(key, Arc::clone(&model));
        Ok(model)
    }

    /// List available models.
    pub fn list_models(&self) -> Vec<String> {
        if !self.model_dir.exists() {
            return Vec::new();
        }

        std::fs::read_dir(&self.model_dir)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().is_dir())
                    .filter_map(|e| e.file_name().into_string().ok())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Unload a model from memory.
    pub fn unload(&self, model_id: &str, version: &str) {
        let key = format!("{}:{}", model_id, version);
        self.models.write().remove(&key);
    }

    /// Clear all loaded models.
    pub fn clear(&self) {
        self.models.write().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_registry_creation() {
        let dir = tempdir().unwrap();
        let registry = ModelRegistry::new(dir.path().to_path_buf());
        assert!(registry.list_models().is_empty());
    }
}
