//! Global application state with thread-safe access.

use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

use crate::data::{
    ChemicalShiftList, ConstraintSet, Molecule, Peak, Spectrum1D, Spectrum2D,
};
use crate::db::Database;

/// The main application state, wrapped for Tauri.
pub struct AppState {
    /// Database connection (Mutex because Connection is not Sync)
    pub db: Arc<Mutex<Option<Database>>>,
    /// Currently loaded molecules
    pub molecules: RwLock<HashMap<Uuid, Molecule>>,
    /// Currently loaded 1D spectra
    pub spectra_1d: RwLock<HashMap<Uuid, Spectrum1D>>,
    /// Currently loaded 2D spectra
    pub spectra_2d: RwLock<HashMap<Uuid, Spectrum2D>>,
    /// Peak lists by spectrum ID
    pub peaks: RwLock<HashMap<Uuid, Vec<Peak>>>,
    /// Chemical shift lists
    pub shift_lists: RwLock<HashMap<Uuid, ChemicalShiftList>>,
    /// Constraint sets
    pub constraints: RwLock<HashMap<Uuid, ConstraintSet>>,
    /// Currently active molecule ID
    pub active_molecule: RwLock<Option<Uuid>>,
    /// Currently active spectrum ID
    pub active_spectrum: RwLock<Option<Uuid>>,
    /// Application data directory
    pub data_dir: PathBuf,
    /// Models directory
    pub models_dir: PathBuf,
}

impl AppState {
    /// Create a new application state.
    pub fn new(data_dir: PathBuf) -> Self {
        let models_dir = data_dir.join("models");

        Self {
            db: Arc::new(Mutex::new(None)),
            molecules: RwLock::new(HashMap::new()),
            spectra_1d: RwLock::new(HashMap::new()),
            spectra_2d: RwLock::new(HashMap::new()),
            peaks: RwLock::new(HashMap::new()),
            shift_lists: RwLock::new(HashMap::new()),
            constraints: RwLock::new(HashMap::new()),
            active_molecule: RwLock::new(None),
            active_spectrum: RwLock::new(None),
            data_dir,
            models_dir,
        }
    }

    /// Initialize database connection.
    pub fn init_database(&self, db_path: &PathBuf) -> crate::error::Result<()> {
        let database = Database::open(db_path)?;
        database.run_migrations()?;
        *self.db.lock() = Some(database);
        Ok(())
    }

    /// Execute a function with database access.
    pub fn with_db<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&Database) -> R,
    {
        let guard = self.db.lock();
        guard.as_ref().map(f)
    }

    /// Execute a function with mutable database access.
    pub fn with_db_mut<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&mut Database) -> R,
    {
        let mut guard = self.db.lock();
        guard.as_mut().map(f)
    }

    // Molecule operations

    /// Add a molecule to the state.
    pub fn add_molecule(&self, molecule: Molecule) -> Uuid {
        let id = molecule.id;
        self.molecules.write().insert(id, molecule);
        id
    }

    /// Get a molecule by ID.
    pub fn get_molecule(&self, id: &Uuid) -> Option<Molecule> {
        self.molecules.read().get(id).cloned()
    }

    /// Set the active molecule.
    pub fn set_active_molecule(&self, id: Option<Uuid>) {
        *self.active_molecule.write() = id;
    }

    /// Get the active molecule.
    pub fn get_active_molecule(&self) -> Option<Molecule> {
        let active_id = *self.active_molecule.read();
        active_id.and_then(|id| self.get_molecule(&id))
    }

    // Spectrum operations

    /// Add a 1D spectrum.
    pub fn add_spectrum_1d(&self, spectrum: Spectrum1D) -> Uuid {
        let id = spectrum.metadata.id;
        self.spectra_1d.write().insert(id, spectrum);
        id
    }

    /// Add a 2D spectrum.
    pub fn add_spectrum_2d(&self, spectrum: Spectrum2D) -> Uuid {
        let id = spectrum.metadata.id;
        self.spectra_2d.write().insert(id, spectrum);
        id
    }

    /// Get a 1D spectrum by ID.
    pub fn get_spectrum_1d(&self, id: &Uuid) -> Option<Spectrum1D> {
        self.spectra_1d.read().get(id).cloned()
    }

    /// Get a 2D spectrum by ID.
    pub fn get_spectrum_2d(&self, id: &Uuid) -> Option<Spectrum2D> {
        self.spectra_2d.read().get(id).cloned()
    }

    /// Set the active spectrum.
    pub fn set_active_spectrum(&self, id: Option<Uuid>) {
        *self.active_spectrum.write() = id;
    }

    // Peak operations

    /// Add peaks for a spectrum.
    pub fn add_peaks(&self, spectrum_id: Uuid, new_peaks: Vec<Peak>) {
        let mut peaks = self.peaks.write();
        peaks
            .entry(spectrum_id)
            .or_insert_with(Vec::new)
            .extend(new_peaks);
    }

    /// Get peaks for a spectrum.
    pub fn get_peaks(&self, spectrum_id: &Uuid) -> Vec<Peak> {
        self.peaks
            .read()
            .get(spectrum_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Clear peaks for a spectrum.
    pub fn clear_peaks(&self, spectrum_id: &Uuid) {
        self.peaks.write().remove(spectrum_id);
    }

    // Chemical shift list operations

    /// Add a chemical shift list.
    pub fn add_shift_list(&self, list: ChemicalShiftList) -> Uuid {
        let id = list.id;
        self.shift_lists.write().insert(id, list);
        id
    }

    /// Get a chemical shift list by ID.
    pub fn get_shift_list(&self, id: &Uuid) -> Option<ChemicalShiftList> {
        self.shift_lists.read().get(id).cloned()
    }

    // Constraint operations

    /// Add a constraint set.
    pub fn add_constraint_set(&self, set: ConstraintSet) -> Uuid {
        let id = set.id;
        self.constraints.write().insert(id, set);
        id
    }

    /// Get a constraint set by ID.
    pub fn get_constraint_set(&self, id: &Uuid) -> Option<ConstraintSet> {
        self.constraints.read().get(id).cloned()
    }

    // Statistics

    /// Get counts of loaded data.
    pub fn stats(&self) -> StateStats {
        StateStats {
            molecules: self.molecules.read().len(),
            spectra_1d: self.spectra_1d.read().len(),
            spectra_2d: self.spectra_2d.read().len(),
            peak_lists: self.peaks.read().len(),
            shift_lists: self.shift_lists.read().len(),
            constraint_sets: self.constraints.read().len(),
        }
    }
}

/// Statistics about the current state.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StateStats {
    pub molecules: usize,
    pub spectra_1d: usize,
    pub spectra_2d: usize,
    pub peak_lists: usize,
    pub shift_lists: usize,
    pub constraint_sets: usize,
}

impl Default for AppState {
    fn default() -> Self {
        Self::new(PathBuf::from("."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Molecule;

    #[test]
    fn test_molecule_operations() {
        let state = AppState::default();

        let mol = Molecule::from_sequence("test", "ACDEF", "A");
        let mol_id = mol.id;

        state.add_molecule(mol);
        state.set_active_molecule(Some(mol_id));

        let retrieved = state.get_molecule(&mol_id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().sequence, "ACDEF");

        let active = state.get_active_molecule();
        assert!(active.is_some());
    }

    #[test]
    fn test_stats() {
        let state = AppState::default();

        let mol = Molecule::from_sequence("test", "ABC", "A");
        state.add_molecule(mol);

        let stats = state.stats();
        assert_eq!(stats.molecules, 1);
        assert_eq!(stats.spectra_1d, 0);
    }
}
