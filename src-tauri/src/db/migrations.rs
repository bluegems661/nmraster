//! Database schema migrations.

use rusqlite::Connection;

use crate::error::{DatabaseError, NmrError, Result};

/// Run all migrations in order.
pub fn run_migrations(conn: &Connection) -> Result<()> {
    // Create migrations table if it doesn't exist
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        )",
        [],
    )
    .map_err(DatabaseError::from)?;

    // Get current version
    let current_version: i32 = conn
        .query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations",
            [],
            |row| row.get(0),
        )
        .map_err(DatabaseError::from)?;

    // Apply migrations
    let migrations = get_migrations();

    for (version, sql) in migrations.iter().enumerate() {
        let version = (version + 1) as i32;
        if version > current_version {
            tracing::info!("Applying migration {}", version);

            conn.execute_batch(sql)
                .map_err(|e| NmrError::Database(DatabaseError::Migration(format!(
                    "Migration {} failed: {}",
                    version, e
                ))))?;

            conn.execute(
                "INSERT INTO schema_migrations (version) VALUES (?1)",
                [version],
            )
            .map_err(DatabaseError::from)?;
        }
    }

    Ok(())
}

/// Get all migration SQL statements.
fn get_migrations() -> Vec<&'static str> {
    vec![
        // Migration 1: Core tables
        r#"
        -- Molecules
        CREATE TABLE molecules (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            sequence_length INTEGER NOT NULL,
            sequence TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        -- Chains
        CREATE TABLE chains (
            id TEXT PRIMARY KEY,
            molecule_id TEXT NOT NULL REFERENCES molecules(id) ON DELETE CASCADE,
            chain_code TEXT NOT NULL,
            polymer_type TEXT NOT NULL,
            start_residue INTEGER NOT NULL,
            end_residue INTEGER NOT NULL
        );
        CREATE INDEX idx_chains_molecule ON chains(molecule_id);

        -- Residues
        CREATE TABLE residues (
            id TEXT PRIMARY KEY,
            molecule_id TEXT NOT NULL REFERENCES molecules(id) ON DELETE CASCADE,
            chain_id TEXT NOT NULL REFERENCES chains(id) ON DELETE CASCADE,
            sequence_code INTEGER NOT NULL,
            residue_name TEXT NOT NULL,
            one_letter_code TEXT NOT NULL
        );
        CREATE INDEX idx_residues_molecule ON residues(molecule_id);
        CREATE INDEX idx_residues_chain ON residues(chain_id);
        CREATE UNIQUE INDEX idx_residues_seq ON residues(molecule_id, chain_id, sequence_code);

        -- Atoms
        CREATE TABLE atoms (
            id TEXT PRIMARY KEY,
            residue_id TEXT NOT NULL REFERENCES residues(id) ON DELETE CASCADE,
            atom_name TEXT NOT NULL,
            element TEXT NOT NULL,
            isotope_number INTEGER
        );
        CREATE INDEX idx_atoms_residue ON atoms(residue_id);
        "#,
        // Migration 2: Experiment and spectrum tables
        r#"
        -- Experiments
        CREATE TABLE experiments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            experiment_type TEXT NOT NULL,
            date TEXT NOT NULL,
            spectrometer_frequency_mhz REAL NOT NULL,
            temperature_k REAL,
            ph REAL,
            sample_conditions TEXT,
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        -- Spectra
        CREATE TABLE spectra (
            id TEXT PRIMARY KEY,
            experiment_id TEXT REFERENCES experiments(id) ON DELETE SET NULL,
            name TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            experiment_type TEXT NOT NULL,
            nucleus_types TEXT NOT NULL,  -- JSON array
            spectral_width_hz TEXT NOT NULL,  -- JSON array
            carrier_offset_ppm TEXT NOT NULL,  -- JSON array
            spectrometer_frequency_mhz REAL NOT NULL,
            original_points TEXT NOT NULL,  -- JSON array
            processed_points TEXT NOT NULL,  -- JSON array
            file_path TEXT,
            is_processed INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX idx_spectra_experiment ON spectra(experiment_id);
        "#,
        // Migration 3: Peak and assignment tables
        r#"
        -- Peaks
        CREATE TABLE peaks (
            id TEXT PRIMARY KEY,
            spectrum_id TEXT NOT NULL REFERENCES spectra(id) ON DELETE CASCADE,
            position_ppm TEXT NOT NULL,  -- JSON array
            position_error TEXT,  -- JSON array
            intensity REAL NOT NULL,
            volume REAL,
            line_width_hz TEXT,  -- JSON array
            signal_noise_ratio REAL,
            is_artifact INTEGER NOT NULL DEFAULT 0,
            figure_of_merit REAL DEFAULT 1.0,
            annotation TEXT
        );
        CREATE INDEX idx_peaks_spectrum ON peaks(spectrum_id);

        -- Chemical shift lists
        CREATE TABLE chemical_shift_lists (
            id TEXT PRIMARY KEY,
            molecule_id TEXT NOT NULL REFERENCES molecules(id) ON DELETE CASCADE,
            experiment_id TEXT REFERENCES experiments(id) ON DELETE SET NULL,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX idx_shift_lists_molecule ON chemical_shift_lists(molecule_id);

        -- Chemical shifts
        CREATE TABLE chemical_shifts (
            id TEXT PRIMARY KEY,
            list_id TEXT NOT NULL REFERENCES chemical_shift_lists(id) ON DELETE CASCADE,
            atom_id TEXT NOT NULL REFERENCES atoms(id) ON DELETE CASCADE,
            value REAL NOT NULL,
            error REAL,
            ambiguity_code INTEGER DEFAULT 1,
            confidence REAL DEFAULT 1.0
        );
        CREATE INDEX idx_shifts_list ON chemical_shifts(list_id);
        CREATE INDEX idx_shifts_atom ON chemical_shifts(atom_id);
        CREATE INDEX idx_shifts_value ON chemical_shifts(value);
        CREATE UNIQUE INDEX idx_shifts_unique ON chemical_shifts(list_id, atom_id);

        -- Peak assignments
        CREATE TABLE peak_assignments (
            id TEXT PRIMARY KEY,
            peak_id TEXT NOT NULL REFERENCES peaks(id) ON DELETE CASCADE,
            dimension_index INTEGER NOT NULL,
            atom_id TEXT REFERENCES atoms(id) ON DELETE SET NULL,
            probability REAL DEFAULT 1.0,
            is_primary INTEGER DEFAULT 1
        );
        CREATE INDEX idx_assignments_peak ON peak_assignments(peak_id);
        CREATE INDEX idx_assignments_atom ON peak_assignments(atom_id);
        "#,
        // Migration 4: Constraint tables
        r#"
        -- Constraint sets
        CREATE TABLE constraint_sets (
            id TEXT PRIMARY KEY,
            molecule_id TEXT NOT NULL REFERENCES molecules(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX idx_constraint_sets_molecule ON constraint_sets(molecule_id);

        -- Distance constraints
        CREATE TABLE distance_constraints (
            id TEXT PRIMARY KEY,
            constraint_set_id TEXT NOT NULL REFERENCES constraint_sets(id) ON DELETE CASCADE,
            atom1_id TEXT NOT NULL REFERENCES atoms(id) ON DELETE CASCADE,
            atom2_id TEXT NOT NULL REFERENCES atoms(id) ON DELETE CASCADE,
            lower_bound REAL NOT NULL,
            upper_bound REAL NOT NULL,
            target REAL NOT NULL,
            weight REAL DEFAULT 1.0,
            peak_id TEXT REFERENCES peaks(id) ON DELETE SET NULL,
            constraint_type TEXT NOT NULL,
            ambiguity INTEGER DEFAULT 1
        );
        CREATE INDEX idx_distance_constraints_set ON distance_constraints(constraint_set_id);
        CREATE INDEX idx_distance_constraints_atom1 ON distance_constraints(atom1_id);
        CREATE INDEX idx_distance_constraints_atom2 ON distance_constraints(atom2_id);

        -- Dihedral constraints
        CREATE TABLE dihedral_constraints (
            id TEXT PRIMARY KEY,
            constraint_set_id TEXT NOT NULL REFERENCES constraint_sets(id) ON DELETE CASCADE,
            atom1_id TEXT NOT NULL REFERENCES atoms(id) ON DELETE CASCADE,
            atom2_id TEXT NOT NULL REFERENCES atoms(id) ON DELETE CASCADE,
            atom3_id TEXT NOT NULL REFERENCES atoms(id) ON DELETE CASCADE,
            atom4_id TEXT NOT NULL REFERENCES atoms(id) ON DELETE CASCADE,
            target_angle REAL NOT NULL,
            tolerance REAL NOT NULL,
            weight REAL DEFAULT 1.0,
            source TEXT NOT NULL
        );
        CREATE INDEX idx_dihedral_constraints_set ON dihedral_constraints(constraint_set_id);
        "#,
        // Migration 5: Relaxation data tables
        r#"
        -- Relaxation data
        CREATE TABLE relaxation_data (
            id TEXT PRIMARY KEY,
            molecule_id TEXT NOT NULL REFERENCES molecules(id) ON DELETE CASCADE,
            residue_seq_code INTEGER NOT NULL,
            atom_name TEXT NOT NULL,
            data_type TEXT NOT NULL,
            field_strength_mhz REAL NOT NULL,
            temperature_k REAL NOT NULL,
            value REAL NOT NULL,
            error REAL NOT NULL
        );
        CREATE INDEX idx_relaxation_molecule ON relaxation_data(molecule_id);
        CREATE INDEX idx_relaxation_residue ON relaxation_data(residue_seq_code);

        -- CPMG dispersion data
        CREATE TABLE cpmg_dispersion (
            id TEXT PRIMARY KEY,
            molecule_id TEXT NOT NULL REFERENCES molecules(id) ON DELETE CASCADE,
            residue_seq_code INTEGER NOT NULL,
            atom_name TEXT NOT NULL,
            field_strength_mhz REAL NOT NULL,
            temperature_k REAL NOT NULL,
            dispersion_points TEXT NOT NULL,  -- JSON array of (nu, r2eff, error)
            fitted_parameters TEXT  -- JSON object
        );
        CREATE INDEX idx_cpmg_molecule ON cpmg_dispersion(molecule_id);
        CREATE INDEX idx_cpmg_residue ON cpmg_dispersion(residue_seq_code);
        "#,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migrations_run_successfully() {
        let conn = Connection::open_in_memory().unwrap();
        let result = run_migrations(&conn);
        assert!(result.is_ok());
    }

    #[test]
    fn test_migrations_are_idempotent() {
        let conn = Connection::open_in_memory().unwrap();

        // Run twice
        run_migrations(&conn).unwrap();
        let result = run_migrations(&conn);

        assert!(result.is_ok());
    }

    #[test]
    fn test_tables_created() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        // Check that core tables exist
        let tables = vec![
            "molecules",
            "chains",
            "residues",
            "atoms",
            "experiments",
            "spectra",
            "peaks",
            "chemical_shift_lists",
            "chemical_shifts",
            "peak_assignments",
            "constraint_sets",
            "distance_constraints",
        ];

        for table in tables {
            let exists: i32 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                    [table],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(exists, 1, "Table {} should exist", table);
        }
    }
}
