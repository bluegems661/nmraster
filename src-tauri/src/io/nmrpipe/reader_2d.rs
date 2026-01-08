//! NMRPipe 2D spectrum reader.

use std::fs::File;
use std::io::{BufReader, Seek, SeekFrom};
use std::path::Path;

use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use ndarray::Array2;
use uuid::Uuid;

use super::header::{NmrPipeHeader, HEADER_SIZE};
use crate::data::spectrum::{Spectrum2D, SpectrumMetadata};
use crate::io::error::{IoError, IoResult};

/// Import a 2D NMRPipe spectrum (.ft2).
pub fn import_nmrpipe_2d(path: &str, name: Option<&str>) -> IoResult<Spectrum2D> {
    let path = Path::new(path);

    if !path.exists() {
        return Err(IoError::FileNotFound {
            path: path.display().to_string(),
        });
    }

    // Parse header
    let header = NmrPipeHeader::from_file(path)?;

    // Verify 2D
    if header.ndim < 2 {
        return Err(IoError::InvalidDataFormat {
            reason: format!("Expected 2D spectrum, got {} dimensions", header.ndim),
        });
    }

    // Read data
    let n_f2 = header.size[0]; // Direct dimension
    let n_f1 = header.size[1]; // Indirect dimension

    if n_f1 == 0 || n_f2 == 0 {
        return Err(IoError::InvalidDataFormat {
            reason: format!("Invalid dimensions: {} x {}", n_f1, n_f2),
        });
    }

    let data = read_nmrpipe_data_2d(path, &header, n_f1, n_f2)?;

    // Build metadata
    let experiment_type = header.infer_experiment_type();
    let nuc_f2 = header.infer_nucleus(0);
    let nuc_f1 = header.infer_nucleus(1);

    let spectrum_name = name
        .map(|s| s.to_string())
        .or_else(|| {
            if !header.title.is_empty() {
                Some(header.title.clone())
            } else {
                None
            }
        })
        .unwrap_or_else(|| {
            path.file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "NMRPipe 2D".to_string())
        });

    let metadata = SpectrumMetadata {
        id: Uuid::new_v4(),
        name: spectrum_name,
        experiment_type,
        nucleus_types: vec![nuc_f1, nuc_f2], // F1 (indirect), F2 (direct)
        spectral_width_hz: vec![header.sw[1], header.sw[0]],
        carrier_offset_ppm: vec![header.orig_ppm(1), header.orig_ppm(0)],
        spectrometer_frequency_mhz: header.obs[0],
        original_points: vec![n_f1, n_f2],
        processed_points: vec![n_f1, n_f2],
    };

    // Create spectrum with auto-calculated PPM axes
    let mut spectrum = Spectrum2D::new(data, metadata);
    spectrum.is_processed = header.ftflag[0] && header.ftflag[1];

    // Override with NMRPipe's origin values
    let sw_ppm_f1 = header.sw_ppm(1);
    let sw_ppm_f2 = header.sw_ppm(0);
    let center_ppm_f1 = header.orig_ppm(1);
    let center_ppm_f2 = header.orig_ppm(0);

    spectrum.ppm_axis_f1 = ndarray::Array1::linspace(
        center_ppm_f1 + sw_ppm_f1 / 2.0,
        center_ppm_f1 - sw_ppm_f1 / 2.0,
        n_f1,
    );
    spectrum.ppm_axis_f2 = ndarray::Array1::linspace(
        center_ppm_f2 + sw_ppm_f2 / 2.0,
        center_ppm_f2 - sw_ppm_f2 / 2.0,
        n_f2,
    );

    Ok(spectrum)
}

/// Read float32 2D data from NMRPipe file.
///
/// NMRPipe 2D files store data as: [plane_header][row0][row1]...[rowN]
/// For single-file 2D: header followed by F1*F2 floats (row-major, F1 varies slowest)
fn read_nmrpipe_data_2d(
    path: &Path,
    header: &NmrPipeHeader,
    n_f1: usize,
    n_f2: usize,
) -> IoResult<Array2<f64>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Skip main header
    reader.seek(SeekFrom::Start(HEADER_SIZE as u64))?;

    let mut data = Array2::zeros((n_f1, n_f2));

    // NMRPipe stores data row by row (F1 is the slow dimension)
    // Each row has n_f2 float32 values
    for i in 0..n_f1 {
        for j in 0..n_f2 {
            let value = if header.is_big_endian {
                reader.read_f32::<BigEndian>()
            } else {
                reader.read_f32::<LittleEndian>()
            }
            .map_err(|e| IoError::BinaryReadError {
                reason: format!("Error reading at ({}, {}): {}", i, j, e),
            })?;

            data[[i, j]] = value as f64;
        }
    }

    Ok(data)
}
