//! Bruker 2D spectrum reader.
//!
//! Reads processed 2D spectra from pdata/N/2rr files.

use std::fs::File;
use std::io::{Read, BufReader};
use std::path::Path;

use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use ndarray::Array2;
use uuid::Uuid;

use super::{BrukerAcqParams, BrukerProcParams, BrukerImportOptions};
use crate::data::spectrum::{Spectrum2D, SpectrumMetadata};
use crate::io::error::{IoError, IoResult};

/// Import a Bruker 2D spectrum from pdata directory.
pub fn import_bruker_2d(options: &BrukerImportOptions) -> IoResult<Spectrum2D> {
    let exp_dir = Path::new(&options.path);

    if !exp_dir.exists() {
        return Err(IoError::FileNotFound {
            path: options.path.clone(),
        });
    }

    // Load acquisition parameters
    let acq_params = BrukerAcqParams::load(exp_dir)?;

    // Verify this is actually a 2D experiment
    if acq_params.parmode < 1 {
        return Err(IoError::InvalidDataFormat {
            reason: "Not a 2D experiment (PARMODE < 1)".to_string(),
        });
    }

    let pdata_dir = exp_dir.join("pdata").join(options.procno.to_string());

    if !pdata_dir.exists() {
        return Err(IoError::InvalidBrukerStructure {
            reason: format!("pdata/{} directory not found", options.procno),
        });
    }

    // Load processing parameters
    let proc_params = BrukerProcParams::load(&pdata_dir)?;

    // Get F1 dimension parameters
    let si_f1 = proc_params.si_f1.ok_or_else(|| IoError::MissingParameter {
        param: "SI (F1 dimension)".to_string(),
    })?;

    let sw_p_f1 = proc_params.sw_p_f1.ok_or_else(|| IoError::MissingParameter {
        param: "SW_p (F1 dimension)".to_string(),
    })?;

    let offset_f1 = proc_params.offset_f1.unwrap_or(0.0);
    let sf_f1 = proc_params.sf_f1.unwrap_or(proc_params.sf);

    // Read the 2rr file (real-real quadrant)
    let data_path = pdata_dir.join("2rr");
    let data = read_bruker_2rr(
        &data_path,
        si_f1,
        proc_params.si,
        proc_params.bytordp,
        proc_params.nc_proc,
    )?;

    // Build metadata
    let experiment_type = acq_params.infer_experiment_type();
    let nuc1 = BrukerAcqParams::parse_nucleus(&acq_params.nuc1);
    let nuc2 = acq_params
        .nuc2
        .as_ref()
        .map(|n| BrukerAcqParams::parse_nucleus(n))
        .unwrap_or(nuc1);

    let name = options
        .name
        .clone()
        .or_else(|| acq_params.title.clone())
        .unwrap_or_else(|| format!("{}/{} 2D", acq_params.nuc1, acq_params.nuc2.as_deref().unwrap_or("?")));

    let metadata = SpectrumMetadata {
        id: Uuid::new_v4(),
        name,
        experiment_type,
        nucleus_types: vec![nuc2, nuc1], // F1 (indirect), F2 (direct)
        spectral_width_hz: vec![sw_p_f1 * sf_f1, proc_params.sw_p * proc_params.sf],
        carrier_offset_ppm: vec![
            offset_f1 - sw_p_f1 / 2.0,
            proc_params.offset - proc_params.sw_p / 2.0,
        ],
        spectrometer_frequency_mhz: proc_params.sf,
        original_points: vec![
            acq_params.td_f1.unwrap_or(si_f1) / 2,
            acq_params.td / 2,
        ],
        processed_points: vec![si_f1, proc_params.si],
    };

    // Create spectrum with auto-calculated PPM axes
    let mut spectrum = Spectrum2D::new(data, metadata);
    spectrum.is_processed = true;

    // Override PPM axes with Bruker's OFFSET values
    // Bruker: OFFSET is the ppm of the leftmost (highest) point
    spectrum.ppm_axis_f1 = ndarray::Array1::linspace(
        offset_f1,
        offset_f1 - sw_p_f1,
        si_f1,
    );
    spectrum.ppm_axis_f2 = ndarray::Array1::linspace(
        proc_params.offset,
        proc_params.offset - proc_params.sw_p,
        proc_params.si,
    );

    Ok(spectrum)
}

/// Read Bruker 2rr file (2D processed real-real data).
///
/// Bruker stores 2D data in submatrices. The 2rr file contains
/// SI(F2) * SI(F1) int32 values.
fn read_bruker_2rr(
    path: &Path,
    si_f1: usize,
    si_f2: usize,
    bytordp: u8,
    nc_proc: i32,
) -> IoResult<Array2<f64>> {
    let file = File::open(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            IoError::FileNotFound {
                path: path.display().to_string(),
            }
        } else {
            IoError::Io(e)
        }
    })?;

    let expected_size = si_f1 * si_f2 * 4;
    let actual_size = file.metadata()?.len() as usize;

    if actual_size < expected_size {
        return Err(IoError::InvalidDataFormat {
            reason: format!(
                "2rr file too small: expected {} bytes for {}x{}, got {}",
                expected_size, si_f1, si_f2, actual_size
            ),
        });
    }

    let mut reader = BufReader::new(file);
    let scale = 2.0_f64.powi(nc_proc);

    // Read data row by row
    // Note: Bruker may use submatrix storage, but for most cases
    // the data is stored linearly: F1[0]F2[0..n], F1[1]F2[0..n], ...
    let mut data = Array2::zeros((si_f1, si_f2));

    for i in 0..si_f1 {
        for j in 0..si_f2 {
            let value = if bytordp == 0 {
                reader.read_i32::<LittleEndian>()
            } else {
                reader.read_i32::<BigEndian>()
            }
            .map_err(|e| IoError::BinaryReadError {
                reason: format!("Error reading 2rr at ({}, {}): {}", i, j, e),
            })?;

            data[[i, j]] = value as f64 * scale;
        }
    }

    Ok(data)
}
