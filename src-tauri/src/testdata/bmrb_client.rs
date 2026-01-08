//! BMRB REST API client for fetching deposited chemical shifts.
//!
//! This module provides functions to fetch chemical shift data from specific
//! BMRB entries for testing the assignment algorithm against real data.

use reqwest::blocking::Client;
use std::collections::HashMap;
use thiserror::Error;

/// Error type for BMRB API operations.
#[derive(Debug, Error)]
pub enum BmrbError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Failed to parse response: {0}")]
    Parse(String),
    #[error("Entry {0} not found")]
    NotFound(u32),
    #[error("No chemical shifts in entry {0}")]
    NoShifts(u32),
    #[error("Invalid residue range: {0}")]
    InvalidRange(String),
}

/// A deposited chemical shift from a BMRB entry.
#[derive(Debug, Clone)]
pub struct DepositedShift {
    /// Residue sequence position (1-indexed)
    pub residue_seq_code: i32,
    /// Three-letter residue name (e.g., "ALA", "THR")
    pub residue_name: String,
    /// Atom name (e.g., "CA", "CB", "H", "N")
    pub atom_name: String,
    /// Chemical shift value in ppm
    pub shift_value: f64,
}

/// Create an HTTP client with appropriate headers.
fn create_client() -> Result<Client, BmrbError> {
    Client::builder()
        .user_agent("NMRaster/1.0 (Test Assignment CLI)")
        .build()
        .map_err(BmrbError::Http)
}

/// Fetch deposited chemical shifts for a BMRB entry.
///
/// # Arguments
/// * `entry_id` - BMRB entry ID (e.g., 4493)
///
/// # Returns
/// Vector of deposited shifts with residue and atom information.
pub fn fetch_entry_shifts(entry_id: u32) -> Result<Vec<DepositedShift>, BmrbError> {
    let client = create_client()?;
    // Fetch the full entry - chemical shifts are embedded in saveframes
    let url = format!("https://api.bmrb.io/v2/entry/{}", entry_id);

    let response = client.get(&url).send()?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(BmrbError::NotFound(entry_id));
    }

    let data: serde_json::Value = response.json()?;

    // Navigate to the entry data
    let entry_key = entry_id.to_string();
    let entry = data
        .get(&entry_key)
        .ok_or_else(|| BmrbError::Parse("Entry not found in response".into()))?;

    let saveframes = entry
        .get("saveframes")
        .and_then(|v| v.as_array())
        .ok_or_else(|| BmrbError::Parse("No saveframes found".into()))?;

    // Find the chemical shift saveframe (usually named "chem_shift_set_1" or similar)
    let shift_saveframe = saveframes
        .iter()
        .find(|sf| {
            sf.get("category")
                .and_then(|v| v.as_str())
                .map(|c| c == "assigned_chemical_shifts")
                .unwrap_or(false)
        })
        .ok_or_else(|| BmrbError::NoShifts(entry_id))?;

    // Find the Atom_chem_shift loop
    let loops = shift_saveframe
        .get("loops")
        .and_then(|v| v.as_array())
        .ok_or_else(|| BmrbError::Parse("No loops in shift saveframe".into()))?;

    let shift_loop = loops
        .iter()
        .find(|l| {
            l.get("category")
                .and_then(|v| v.as_str())
                .map(|c| c == "_Atom_chem_shift")
                .unwrap_or(false)
        })
        .ok_or_else(|| BmrbError::Parse("No Atom_chem_shift loop found".into()))?;

    let tags = shift_loop
        .get("tags")
        .and_then(|v| v.as_array())
        .ok_or_else(|| BmrbError::Parse("No tags in shift loop".into()))?;

    let data_rows = shift_loop
        .get("data")
        .and_then(|v| v.as_array())
        .ok_or_else(|| BmrbError::Parse("No data in shift loop".into()))?;

    // Find column indices
    let seq_idx = tags
        .iter()
        .position(|t| t.as_str() == Some("Seq_ID"))
        .ok_or_else(|| BmrbError::Parse("Missing Seq_ID column".into()))?;
    let comp_idx = tags
        .iter()
        .position(|t| t.as_str() == Some("Comp_ID"))
        .ok_or_else(|| BmrbError::Parse("Missing Comp_ID column".into()))?;
    let atom_idx = tags
        .iter()
        .position(|t| t.as_str() == Some("Atom_ID"))
        .ok_or_else(|| BmrbError::Parse("Missing Atom_ID column".into()))?;
    let val_idx = tags
        .iter()
        .position(|t| t.as_str() == Some("Val"))
        .ok_or_else(|| BmrbError::Parse("Missing Val column".into()))?;

    // Parse rows
    let mut shifts = Vec::new();
    for row in data_rows {
        let row_arr = row.as_array().ok_or_else(|| BmrbError::Parse("Invalid row".into()))?;

        let seq_code = row_arr
            .get(seq_idx)
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<i32>().ok());

        let comp_id = row_arr
            .get(comp_idx)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let atom_id = row_arr
            .get(atom_idx)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let shift_val = row_arr
            .get(val_idx)
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok());

        if let (Some(seq), Some(comp), Some(atom), Some(val)) =
            (seq_code, comp_id, atom_id, shift_val)
        {
            shifts.push(DepositedShift {
                residue_seq_code: seq,
                residue_name: comp,
                atom_name: atom,
                shift_value: val,
            });
        }
    }

    if shifts.is_empty() {
        return Err(BmrbError::NoShifts(entry_id));
    }

    Ok(shifts)
}

/// Extract sequence from deposited shifts.
///
/// Returns a one-letter amino acid sequence derived from the shifts.
pub fn extract_sequence(shifts: &[DepositedShift]) -> String {
    // Group by sequence code and get residue names
    let mut residues: HashMap<i32, String> = HashMap::new();
    for shift in shifts {
        residues
            .entry(shift.residue_seq_code)
            .or_insert_with(|| shift.residue_name.clone());
    }

    // Sort by sequence code and convert to one-letter codes
    let mut seq_codes: Vec<_> = residues.keys().copied().collect();
    seq_codes.sort();

    seq_codes
        .iter()
        .filter_map(|seq| residues.get(seq))
        .map(|name| three_to_one(name))
        .collect()
}

/// Filter shifts to a residue range.
///
/// # Arguments
/// * `shifts` - All deposited shifts
/// * `range` - Range string like "45-62" (1-indexed, inclusive)
pub fn filter_residue_range(
    shifts: &[DepositedShift],
    range: &str,
) -> Result<Vec<DepositedShift>, BmrbError> {
    let parts: Vec<&str> = range.split('-').collect();
    if parts.len() != 2 {
        return Err(BmrbError::InvalidRange(range.to_string()));
    }

    let start: i32 = parts[0]
        .trim()
        .parse()
        .map_err(|_| BmrbError::InvalidRange(range.to_string()))?;
    let end: i32 = parts[1]
        .trim()
        .parse()
        .map_err(|_| BmrbError::InvalidRange(range.to_string()))?;

    Ok(shifts
        .iter()
        .filter(|s| s.residue_seq_code >= start && s.residue_seq_code <= end)
        .cloned()
        .collect())
}

/// Renumber shifts to start from 1.
///
/// Useful when working with a subset of residues.
pub fn renumber_shifts(shifts: &mut [DepositedShift]) {
    if shifts.is_empty() {
        return;
    }

    let min_seq = shifts.iter().map(|s| s.residue_seq_code).min().unwrap_or(1);
    let offset = min_seq - 1;

    for shift in shifts {
        shift.residue_seq_code -= offset;
    }
}

/// Convert three-letter amino acid code to one-letter code.
pub fn three_to_one(three: &str) -> char {
    match three.to_uppercase().as_str() {
        "ALA" => 'A',
        "ARG" => 'R',
        "ASN" => 'N',
        "ASP" => 'D',
        "CYS" => 'C',
        "GLN" => 'Q',
        "GLU" => 'E',
        "GLY" => 'G',
        "HIS" => 'H',
        "ILE" => 'I',
        "LEU" => 'L',
        "LYS" => 'K',
        "MET" => 'M',
        "PHE" => 'F',
        "PRO" => 'P',
        "SER" => 'S',
        "THR" => 'T',
        "TRP" => 'W',
        "TYR" => 'Y',
        "VAL" => 'V',
        _ => 'X',
    }
}

/// Convert one-letter amino acid code to three-letter code.
pub fn one_to_three(one: char) -> &'static str {
    match one.to_ascii_uppercase() {
        'A' => "ALA",
        'R' => "ARG",
        'N' => "ASN",
        'D' => "ASP",
        'C' => "CYS",
        'Q' => "GLN",
        'E' => "GLU",
        'G' => "GLY",
        'H' => "HIS",
        'I' => "ILE",
        'L' => "LEU",
        'K' => "LYS",
        'M' => "MET",
        'F' => "PHE",
        'P' => "PRO",
        'S' => "SER",
        'T' => "THR",
        'W' => "TRP",
        'Y' => "TYR",
        'V' => "VAL",
        _ => "UNK",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_three_to_one() {
        assert_eq!(three_to_one("ALA"), 'A');
        assert_eq!(three_to_one("ala"), 'A');
        assert_eq!(three_to_one("THR"), 'T');
        assert_eq!(three_to_one("UNK"), 'X');
    }

    #[test]
    fn test_one_to_three() {
        assert_eq!(one_to_three('A'), "ALA");
        assert_eq!(one_to_three('t'), "THR");
        assert_eq!(one_to_three('X'), "UNK");
    }

    #[test]
    fn test_filter_residue_range() {
        let shifts = vec![
            DepositedShift {
                residue_seq_code: 1,
                residue_name: "ALA".into(),
                atom_name: "CA".into(),
                shift_value: 53.0,
            },
            DepositedShift {
                residue_seq_code: 2,
                residue_name: "THR".into(),
                atom_name: "CA".into(),
                shift_value: 62.0,
            },
            DepositedShift {
                residue_seq_code: 3,
                residue_name: "GLY".into(),
                atom_name: "CA".into(),
                shift_value: 45.0,
            },
        ];

        let filtered = filter_residue_range(&shifts, "1-2").unwrap();
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].residue_name, "ALA");
        assert_eq!(filtered[1].residue_name, "THR");
    }

    #[test]
    fn test_extract_sequence() {
        let shifts = vec![
            DepositedShift {
                residue_seq_code: 1,
                residue_name: "ALA".into(),
                atom_name: "CA".into(),
                shift_value: 53.0,
            },
            DepositedShift {
                residue_seq_code: 1,
                residue_name: "ALA".into(),
                atom_name: "CB".into(),
                shift_value: 19.0,
            },
            DepositedShift {
                residue_seq_code: 2,
                residue_name: "CYS".into(),
                atom_name: "CA".into(),
                shift_value: 58.0,
            },
        ];

        let seq = extract_sequence(&shifts);
        assert_eq!(seq, "AC");
    }
}
