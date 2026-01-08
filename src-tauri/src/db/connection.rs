//! SQLite database connection with WAL mode.

use rusqlite::{Connection, OpenFlags};
use std::path::Path;

use crate::error::{DatabaseError, NmrError, Result};

/// Database wrapper with connection management.
pub struct Database {
    conn: Connection,
}

impl Database {
    /// Open or create a database at the given path.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(DatabaseError::from)?;

        // Enable WAL mode for better concurrency
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(DatabaseError::from)?;

        // Enable foreign keys
        conn.pragma_update(None, "foreign_keys", "ON")
            .map_err(DatabaseError::from)?;

        // Set busy timeout to 5 seconds
        conn.busy_timeout(std::time::Duration::from_secs(5))
            .map_err(DatabaseError::from)?;

        Ok(Self { conn })
    }

    /// Open an in-memory database (for testing).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(DatabaseError::from)?;

        conn.pragma_update(None, "foreign_keys", "ON")
            .map_err(DatabaseError::from)?;

        Ok(Self { conn })
    }

    /// Run all pending migrations.
    pub fn run_migrations(&self) -> Result<()> {
        super::migrations::run_migrations(&self.conn)
    }

    /// Get a reference to the underlying connection.
    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    /// Execute a query that returns no results.
    pub fn execute(&self, sql: &str, params: &[&dyn rusqlite::ToSql]) -> Result<usize> {
        self.conn
            .execute(sql, params)
            .map_err(|e| NmrError::Database(DatabaseError::Sqlite(e)))
    }

    /// Begin a transaction.
    pub fn transaction(&mut self) -> Result<rusqlite::Transaction<'_>> {
        self.conn
            .transaction()
            .map_err(|e| NmrError::Database(DatabaseError::Sqlite(e)))
    }

    /// Get the last inserted row ID.
    pub fn last_insert_rowid(&self) -> i64 {
        self.conn.last_insert_rowid()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_in_memory() {
        let db = Database::open_in_memory();
        assert!(db.is_ok());
    }

    #[test]
    fn test_migrations() {
        let db = Database::open_in_memory().unwrap();
        let result = db.run_migrations();
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute() {
        let db = Database::open_in_memory().unwrap();
        db.run_migrations().unwrap();

        // Test that we can execute a simple query
        let result = db.execute(
            "INSERT INTO molecules (id, name, sequence_length) VALUES (?1, ?2, ?3)",
            &[&"test-id", &"test-molecule", &10i32],
        );

        assert!(result.is_ok());
    }
}
