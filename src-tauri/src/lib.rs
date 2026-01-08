//! NMRaster - Next-generation NMR platform with simultaneous multi-experiment analysis.
//!
//! This application provides tools for processing, analyzing, and assigning NMR spectra
//! using factor graphs and belief propagation for global optimization.

pub mod error;
pub mod data;
pub mod state;
pub mod db;
pub mod processing;
pub mod inference;
pub mod ml;
pub mod commands;
pub mod testdata;
pub mod io;

use std::path::PathBuf;
use tauri::Manager;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use commands::{
    // Spectrum commands
    load_spectrum_1d,
    get_spectrum_1d,
    process_spectrum_1d,
    list_spectra,
    get_spectrum_peaks,
    // Assignment commands
    load_molecule_from_sequence,
    get_active_molecule,
    get_molecule_residues,
    get_residue_atoms,
    create_shift_list,
    add_chemical_shift,
    get_shifts_for_residue,
    list_shift_lists,
    run_assignment,
    run_real_assignment,
    // Database commands
    init_database,
    save_molecule_to_db,
    load_molecule_from_db,
    list_molecules_in_db,
    delete_molecule_from_db,
    get_db_stats,
    get_app_stats,
    // Analysis commands
    pick_peaks_1d,
    integrate_peak,
    clear_spectrum_peaks,
    // Test data commands
    generate_test_data,
    generate_demo_2d,
    get_spectrum_2d,
    get_bmrb_stats,
    get_bmrb_shift,
    get_expected_peaks,
    create_shift_list_from_ground_truth,
    // Import commands
    import_bruker_1d,
    import_bruker_2d,
    import_nmrpipe_1d,
    import_nmrpipe_2d,
    import_spectrum_auto,
};
use state::AppState;

/// Initialize tracing/logging.
fn init_tracing() {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();
}

/// Application entry point.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    init_tracing();

    tracing::info!("Starting NMRaster application");

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .setup(|app| {
            // Get app data directory
            let app_data_dir = app
                .path()
                .app_data_dir()
                .unwrap_or_else(|_| PathBuf::from("."));

            // Create data directory if it doesn't exist
            std::fs::create_dir_all(&app_data_dir).ok();

            // Initialize application state
            let state = AppState::new(app_data_dir.clone());

            // Initialize database
            let db_path = app_data_dir.join("nmraster.db");
            if let Err(e) = state.init_database(&db_path) {
                tracing::error!("Failed to initialize database: {}", e);
            } else {
                tracing::info!("Database initialized at {:?}", db_path);
            }

            app.manage(state);

            tracing::info!("NMRaster setup complete");
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // Spectrum
            load_spectrum_1d,
            get_spectrum_1d,
            process_spectrum_1d,
            list_spectra,
            get_spectrum_peaks,
            // Assignment
            load_molecule_from_sequence,
            get_active_molecule,
            get_molecule_residues,
            get_residue_atoms,
            create_shift_list,
            add_chemical_shift,
            get_shifts_for_residue,
            list_shift_lists,
            run_assignment,
            run_real_assignment,
            // Database
            init_database,
            save_molecule_to_db,
            load_molecule_from_db,
            list_molecules_in_db,
            delete_molecule_from_db,
            get_db_stats,
            get_app_stats,
            // Analysis
            pick_peaks_1d,
            integrate_peak,
            clear_spectrum_peaks,
            // Test data
            generate_test_data,
            generate_demo_2d,
            get_spectrum_2d,
            get_bmrb_stats,
            get_bmrb_shift,
            get_expected_peaks,
            create_shift_list_from_ground_truth,
            // Import
            import_bruker_1d,
            import_bruker_2d,
            import_nmrpipe_1d,
            import_nmrpipe_2d,
            import_spectrum_auto,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
