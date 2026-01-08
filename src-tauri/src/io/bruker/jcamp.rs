//! JCAMP-DX parameter file parser for Bruker acqus/procs files.
//!
//! JCAMP-DX is a standard format for storing chemical data. Bruker uses a variant
//! with parameters prefixed by ## or $$.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::io::error::{IoError, IoResult};

/// Parsed JCAMP-DX parameters.
#[derive(Debug, Clone)]
pub struct JcampParams {
    /// Scalar parameters (##$PARAM= value)
    params: HashMap<String, String>,
    /// Array parameters (##$PARAM= (0..n) values...)
    arrays: HashMap<String, Vec<String>>,
}

impl JcampParams {
    /// Parse a JCAMP-DX file (acqus, procs, etc.).
    pub fn parse_file(path: &Path) -> IoResult<Self> {
        let content = fs::read_to_string(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                IoError::FileNotFound {
                    path: path.display().to_string(),
                }
            } else {
                IoError::Io(e)
            }
        })?;

        Self::parse_str(&content)
    }

    /// Parse JCAMP-DX content from a string.
    pub fn parse_str(content: &str) -> IoResult<Self> {
        let mut params = HashMap::new();
        let mut arrays = HashMap::new();

        let mut current_param: Option<String> = None;
        let mut current_values: Vec<String> = Vec::new();
        let mut in_array = false;
        let mut array_remaining = 0usize;

        for line in content.lines() {
            let line = line.trim();

            // Skip comments and empty lines
            if line.is_empty() || line.starts_with("$$") {
                continue;
            }

            // Check for parameter definition
            if line.starts_with("##") {
                // Save previous array if we were collecting one
                if let Some(param) = current_param.take() {
                    if !current_values.is_empty() {
                        arrays.insert(param, current_values.clone());
                        current_values.clear();
                    }
                }
                in_array = false;

                // Parse the parameter line
                let line = line.trim_start_matches("##").trim_start_matches('$');

                if let Some(eq_pos) = line.find('=') {
                    let param_name = line[..eq_pos].trim().to_string();
                    let value_part = line[eq_pos + 1..].trim();

                    // Check if this is an array definition: (0..n)
                    if value_part.starts_with('(') {
                        if let Some(end_paren) = value_part.find(')') {
                            let range_str = &value_part[1..end_paren];
                            if let Some(dots) = range_str.find("..") {
                                let end: usize = range_str[dots + 2..]
                                    .trim()
                                    .parse()
                                    .unwrap_or(0);
                                array_remaining = end + 1;
                                in_array = true;
                                current_param = Some(param_name.clone());

                                // Values may start on the same line after the range
                                let after_range = value_part[end_paren + 1..].trim();
                                if !after_range.is_empty() {
                                    Self::parse_values_into(after_range, &mut current_values);
                                }
                            }
                        }
                    } else {
                        // Scalar parameter
                        params.insert(param_name, value_part.to_string());
                    }
                }
            } else if in_array && current_param.is_some() {
                // Continuation of array values
                Self::parse_values_into(line, &mut current_values);
            }
        }

        // Save final array if any
        if let Some(param) = current_param.take() {
            if !current_values.is_empty() {
                arrays.insert(param, current_values);
            }
        }

        Ok(Self { params, arrays })
    }

    /// Parse space/newline separated values into a vector.
    fn parse_values_into(s: &str, values: &mut Vec<String>) {
        for token in s.split_whitespace() {
            // Handle Bruker's angle bracket format: <value>
            let token = token.trim_start_matches('<').trim_end_matches('>');
            if !token.is_empty() {
                values.push(token.to_string());
            }
        }
    }

    /// Get a scalar parameter as a string.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.params.get(key).map(|s| s.as_str())
    }

    /// Get a scalar parameter as f64.
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.params.get(key).and_then(|s| s.parse().ok())
    }

    /// Get a scalar parameter as i64.
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.params.get(key).and_then(|s| s.parse().ok())
    }

    /// Get a scalar parameter as usize.
    pub fn get_usize(&self, key: &str) -> Option<usize> {
        self.params.get(key).and_then(|s| s.parse().ok())
    }

    /// Get an array parameter.
    pub fn get_array(&self, key: &str) -> Option<&[String]> {
        self.arrays.get(key).map(|v| v.as_slice())
    }

    /// Get an array parameter as f64 values.
    pub fn get_array_f64(&self, key: &str) -> Option<Vec<f64>> {
        self.arrays.get(key).map(|arr| {
            arr.iter()
                .filter_map(|s| s.parse::<f64>().ok())
                .collect()
        })
    }

    /// Get required parameter or return error.
    pub fn require_f64(&self, key: &str) -> IoResult<f64> {
        self.get_f64(key).ok_or_else(|| IoError::MissingParameter {
            param: key.to_string(),
        })
    }

    /// Get required parameter or return error.
    pub fn require_usize(&self, key: &str) -> IoResult<usize> {
        self.get_usize(key).ok_or_else(|| IoError::MissingParameter {
            param: key.to_string(),
        })
    }

    /// Check if a parameter exists.
    pub fn has(&self, key: &str) -> bool {
        self.params.contains_key(key) || self.arrays.contains_key(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_scalar_params() {
        let content = r#"
##TITLE= 1H Spectrum
##$BF1= 600.13
##$TD= 65536
##$SW_h= 9615.38461538462
"#;
        let params = JcampParams::parse_str(content).unwrap();

        assert_eq!(params.get_str("TITLE"), Some("1H Spectrum"));
        assert!((params.get_f64("BF1").unwrap() - 600.13).abs() < 0.001);
        assert_eq!(params.get_usize("TD"), Some(65536));
        assert!((params.get_f64("SW_h").unwrap() - 9615.38).abs() < 0.1);
    }

    #[test]
    fn test_parse_array_params() {
        let content = r#"
##$D= (0..63)
0.1 0.2 0.3 0.4
0.5 0.6 0.7 0.8
"#;
        let params = JcampParams::parse_str(content).unwrap();

        let arr = params.get_array("D").unwrap();
        assert!(arr.len() >= 8);
        assert_eq!(arr[0], "0.1");
    }

    #[test]
    fn test_require_missing_param() {
        let content = "##$BF1= 600.13";
        let params = JcampParams::parse_str(content).unwrap();

        assert!(params.require_f64("SW_h").is_err());
    }
}
