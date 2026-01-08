//! Prepared SQL queries for database operations.

use rusqlite::{params, Connection, OptionalExtension};
use uuid::Uuid;

use crate::data::{Atom, Chain, Element, Molecule, PolymerType, Residue};
use crate::error::{DatabaseError, Result};

/// Save a molecule to the database.
pub fn save_molecule(conn: &Connection, mol: &Molecule) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO molecules (id, name, sequence_length, sequence) VALUES (?1, ?2, ?3, ?4)",
        params![
            mol.id.to_string(),
            mol.name,
            mol.num_residues() as i32,
            mol.sequence
        ],
    )
    .map_err(DatabaseError::from)?;

    // Save chains
    for chain in &mol.chains {
        save_chain(conn, chain, &mol.id)?;
    }

    // Save residues and atoms
    for residue in mol.residues() {
        let chain = mol
            .chains
            .iter()
            .find(|c| c.chain_code == residue.chain_code)
            .ok_or_else(|| DatabaseError::NotFound {
                entity_type: "Chain".to_string(),
                id: residue.chain_code.clone(),
            })?;

        save_residue(conn, residue, &mol.id, &chain.id)?;

        // Save atoms for this residue
        for atom in mol.get_atoms_for_residue(residue.sequence_code) {
            save_atom(conn, atom, &residue.id)?;
        }
    }

    Ok(())
}

/// Save a chain to the database.
fn save_chain(conn: &Connection, chain: &Chain, molecule_id: &Uuid) -> Result<()> {
    let polymer_type = match chain.polymer_type {
        PolymerType::Protein => "protein",
        PolymerType::DNA => "dna",
        PolymerType::RNA => "rna",
        PolymerType::Polysaccharide => "polysaccharide",
        PolymerType::Other => "other",
    };

    conn.execute(
        "INSERT OR REPLACE INTO chains (id, molecule_id, chain_code, polymer_type, start_residue, end_residue)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            chain.id.to_string(),
            molecule_id.to_string(),
            chain.chain_code,
            polymer_type,
            chain.start_residue,
            chain.end_residue
        ],
    )
    .map_err(DatabaseError::from)?;

    Ok(())
}

/// Save a residue to the database.
fn save_residue(conn: &Connection, residue: &Residue, molecule_id: &Uuid, chain_id: &Uuid) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO residues (id, molecule_id, chain_id, sequence_code, residue_name, one_letter_code)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            residue.id.to_string(),
            molecule_id.to_string(),
            chain_id.to_string(),
            residue.sequence_code,
            residue.residue_name,
            residue.one_letter_code.to_string()
        ],
    )
    .map_err(DatabaseError::from)?;

    Ok(())
}

/// Save an atom to the database.
fn save_atom(conn: &Connection, atom: &Atom, residue_id: &Uuid) -> Result<()> {
    let element = match atom.element {
        Element::H => "H",
        Element::C => "C",
        Element::N => "N",
        Element::O => "O",
        Element::S => "S",
        Element::P => "P",
        Element::F => "F",
        Element::Other => "X",
    };

    conn.execute(
        "INSERT OR REPLACE INTO atoms (id, residue_id, atom_name, element, isotope_number)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![
            atom.id.to_string(),
            residue_id.to_string(),
            atom.atom_name,
            element,
            atom.isotope_number
        ],
    )
    .map_err(DatabaseError::from)?;

    Ok(())
}

/// Load a molecule from the database.
pub fn load_molecule(conn: &Connection, id: &Uuid) -> Result<Option<Molecule>> {
    let row: Option<(String, String, i32, Option<String>)> = conn
        .query_row(
            "SELECT id, name, sequence_length, sequence FROM molecules WHERE id = ?1",
            [id.to_string()],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .optional()
        .map_err(DatabaseError::from)?;

    let Some((_, name, _, sequence)) = row else {
        return Ok(None);
    };

    // If we have a sequence, rebuild the molecule from it
    if let Some(seq) = sequence {
        // Get chain code from first chain
        let chain_code: Option<String> = conn
            .query_row(
                "SELECT chain_code FROM chains WHERE molecule_id = ?1 LIMIT 1",
                [id.to_string()],
                |row| row.get(0),
            )
            .optional()
            .map_err(DatabaseError::from)?;

        let chain = chain_code.unwrap_or_else(|| "A".to_string());
        let mol = Molecule::from_sequence(&name, &seq, &chain);
        return Ok(Some(mol));
    }

    // Otherwise return a basic molecule
    let mol = Molecule::new(&name);
    Ok(Some(mol))
}

/// Get all molecule IDs and names.
pub fn list_molecules(conn: &Connection) -> Result<Vec<(Uuid, String)>> {
    let mut stmt = conn
        .prepare("SELECT id, name FROM molecules ORDER BY name")
        .map_err(DatabaseError::from)?;

    let rows = stmt
        .query_map([], |row| {
            let id_str: String = row.get(0)?;
            let name: String = row.get(1)?;
            Ok((id_str, name))
        })
        .map_err(DatabaseError::from)?;

    let mut results = Vec::new();
    for row in rows {
        let (id_str, name) = row.map_err(DatabaseError::from)?;
        if let Ok(id) = Uuid::parse_str(&id_str) {
            results.push((id, name));
        }
    }

    Ok(results)
}

/// Delete a molecule and all related data.
pub fn delete_molecule(conn: &Connection, id: &Uuid) -> Result<()> {
    conn.execute("DELETE FROM molecules WHERE id = ?1", [id.to_string()])
        .map_err(DatabaseError::from)?;
    Ok(())
}

/// Count total entities in the database.
pub fn count_entities(conn: &Connection) -> Result<EntityCounts> {
    let molecules: i32 = conn
        .query_row("SELECT COUNT(*) FROM molecules", [], |row| row.get(0))
        .map_err(DatabaseError::from)?;

    let spectra: i32 = conn
        .query_row("SELECT COUNT(*) FROM spectra", [], |row| row.get(0))
        .map_err(DatabaseError::from)?;

    let peaks: i32 = conn
        .query_row("SELECT COUNT(*) FROM peaks", [], |row| row.get(0))
        .map_err(DatabaseError::from)?;

    let shifts: i32 = conn
        .query_row("SELECT COUNT(*) FROM chemical_shifts", [], |row| row.get(0))
        .map_err(DatabaseError::from)?;

    let constraints: i32 = conn
        .query_row("SELECT COUNT(*) FROM distance_constraints", [], |row| row.get(0))
        .map_err(DatabaseError::from)?;

    Ok(EntityCounts {
        molecules: molecules as usize,
        spectra: spectra as usize,
        peaks: peaks as usize,
        chemical_shifts: shifts as usize,
        distance_constraints: constraints as usize,
    })
}

/// Counts of entities in the database.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EntityCounts {
    pub molecules: usize,
    pub spectra: usize,
    pub peaks: usize,
    pub chemical_shifts: usize,
    pub distance_constraints: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::migrations::run_migrations;

    fn setup_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();
        conn
    }

    #[test]
    fn test_save_and_load_molecule() {
        let conn = setup_db();

        let mol = Molecule::from_sequence("test_protein", "ACDEF", "A");
        let mol_id = mol.id;

        save_molecule(&conn, &mol).unwrap();

        let loaded = load_molecule(&conn, &mol_id).unwrap();
        assert!(loaded.is_some());

        let loaded = loaded.unwrap();
        assert_eq!(loaded.sequence, "ACDEF");
    }

    #[test]
    fn test_list_molecules() {
        let conn = setup_db();

        let mol1 = Molecule::from_sequence("protein_a", "ABC", "A");
        let mol2 = Molecule::from_sequence("protein_b", "DEF", "A");

        save_molecule(&conn, &mol1).unwrap();
        save_molecule(&conn, &mol2).unwrap();

        let list = list_molecules(&conn).unwrap();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_count_entities() {
        let conn = setup_db();

        let mol = Molecule::from_sequence("test", "ABC", "A");
        save_molecule(&conn, &mol).unwrap();

        let counts = count_entities(&conn).unwrap();
        assert_eq!(counts.molecules, 1);
    }
}
