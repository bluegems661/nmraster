//! Tauri commands for database operations.

use serde::Serialize;
use std::path::PathBuf;
use tauri::State;
use uuid::Uuid;

use crate::db::queries;
use crate::error::Result;
use crate::state::AppState;

/// Initialize database at a path.
#[tauri::command]
pub async fn init_database(path: String, state: State<'_, AppState>) -> Result<()> {
    let db_path = PathBuf::from(path);
    state.init_database(&db_path)
}

/// Save a molecule to the database.
#[tauri::command]
pub async fn save_molecule_to_db(molecule_id: String, state: State<'_, AppState>) -> Result<()> {
    let uuid = Uuid::parse_str(&molecule_id)
        .map_err(|e| crate::error::NmrError::Internal(e.to_string()))?;

    let molecule = state
        .get_molecule(&uuid)
        .ok_or_else(|| crate::error::NmrError::Internal("Molecule not found".to_string()))?;

    state.with_db(|db| {
        queries::save_molecule(db.connection(), &molecule)
    })
    .ok_or_else(|| crate::error::NmrError::Internal("Database not initialized".to_string()))??;

    Ok(())
}

/// Load a molecule from the database.
#[tauri::command]
pub async fn load_molecule_from_db(
    molecule_id: String,
    state: State<'_, AppState>,
) -> Result<Option<String>> {
    let uuid = Uuid::parse_str(&molecule_id)
        .map_err(|e| crate::error::NmrError::Internal(e.to_string()))?;

    let result = state.with_db(|db| {
        queries::load_molecule(db.connection(), &uuid)
    })
    .ok_or_else(|| crate::error::NmrError::Internal("Database not initialized".to_string()))??;

    if let Some(mol) = result {
        let id = state.add_molecule(mol);
        return Ok(Some(id.to_string()));
    }

    Ok(None)
}

/// List all molecules in the database.
#[tauri::command]
pub async fn list_molecules_in_db(state: State<'_, AppState>) -> Result<Vec<DbMoleculeInfo>> {
    let molecules = state.with_db(|db| {
        queries::list_molecules(db.connection())
    })
    .ok_or_else(|| crate::error::NmrError::Internal("Database not initialized".to_string()))??;

    Ok(molecules
        .into_iter()
        .map(|(id, name)| DbMoleculeInfo {
            id: id.to_string(),
            name,
        })
        .collect())
}

/// Molecule information from database.
#[derive(Debug, Serialize)]
pub struct DbMoleculeInfo {
    pub id: String,
    pub name: String,
}

/// Delete a molecule from the database.
#[tauri::command]
pub async fn delete_molecule_from_db(
    molecule_id: String,
    state: State<'_, AppState>,
) -> Result<()> {
    let uuid = Uuid::parse_str(&molecule_id)
        .map_err(|e| crate::error::NmrError::Internal(e.to_string()))?;

    state.with_db(|db| {
        queries::delete_molecule(db.connection(), &uuid)
    })
    .ok_or_else(|| crate::error::NmrError::Internal("Database not initialized".to_string()))??;

    Ok(())
}

/// Get database statistics.
#[tauri::command]
pub async fn get_db_stats(state: State<'_, AppState>) -> Result<DbStats> {
    let counts = state.with_db(|db| {
        queries::count_entities(db.connection())
    })
    .ok_or_else(|| crate::error::NmrError::Internal("Database not initialized".to_string()))??;

    Ok(DbStats {
        molecules: counts.molecules,
        spectra: counts.spectra,
        peaks: counts.peaks,
        chemical_shifts: counts.chemical_shifts,
        distance_constraints: counts.distance_constraints,
    })
}

/// Database statistics.
#[derive(Debug, Serialize)]
pub struct DbStats {
    pub molecules: usize,
    pub spectra: usize,
    pub peaks: usize,
    pub chemical_shifts: usize,
    pub distance_constraints: usize,
}

/// Get application state statistics.
#[tauri::command]
pub async fn get_app_stats(state: State<'_, AppState>) -> Result<crate::state::StateStats> {
    Ok(state.stats())
}
