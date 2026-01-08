//! NMRPipe 1D spectrum reader.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use ndarray::Array1;
use uuid::Uuid;

use super::header::{NmrPipeHeader, HEADER_SIZE};
use crate::data::spectrum::{Spectrum1D, SpectrumMetadata};
use crate::io::error::{IoError, IoResult};

/// Import a 1D NMRPipe spectrum (.ft1, .fid).
pub fn import_nmrpipe_1d(path: &str, name: Option<&str>) -> IoResult<Spectrum1D> {
    let path = Path::new(path);

    if !path.exists() {
        return Err(IoError::FileNotFound {
            path: path.display().to_string(),
        });
    }

    // Parse header
    let header = NmrPipeHeader::from_file(path)?;

    // Verify 1D
    if header.ndim != 1 {
        return Err(IoError::InvalidDataFormat {
            reason: format!("Expected 1D spectrum, got {} dimensions", header.ndim),
        });
    }

    // Read data
    let n_points = header.size[0];
    let data = read_nmrpipe_data_1d(path, &header, n_points)?;

    // Build metadata
    let experiment_type = header.infer_experiment_type();
    let nucleus = header.infer_nucleus(0);

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
                .unwrap_or_else(|| "NMRPipe 1D".to_string())
        });

    let metadata = SpectrumMetadata {
        id: Uuid::new_v4(),
        name: spectrum_name,
        experiment_type,
        nucleus_types: vec![nucleus],
        spectral_width_hz: vec![header.sw[0]],
        carrier_offset_ppm: vec![header.orig_ppm(0)],
        spectrometer_frequency_mhz: header.obs[0],
        original_points: vec![n_points],
        processed_points: vec![n_points],
    };

    // Calculate PPM axis
    // NMRPipe: First point is at highest ppm (orig + sw/2)
    let sw_ppm = header.sw_ppm(0);
    let center_ppm = header.orig_ppm(0);
    let ppm_axis = Array1::linspace(
        center_ppm + sw_ppm / 2.0,
        center_ppm - sw_ppm / 2.0,
        n_points,
    );

    Ok(Spectrum1D {
        metadata,
        real: data,
        imag: None, // NMRPipe processed data is typically real only
        ppm_axis,
        is_processed: header.ftflag[0],
    })
}

/// Read float32 data from NMRPipe file.
fn read_nmrpipe_data_1d(path: &Path, header: &NmrPipeHeader, n_points: usize) -> IoResult<Array1<f64>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Skip header
    reader.seek(SeekFrom::Start(HEADER_SIZE as u64))?;

    let mut data = Vec::with_capacity(n_points);

    for _ in 0..n_points {
        let value = if header.is_big_endian {
            reader.read_f32::<BigEndian>()
        } else {
            reader.read_f32::<LittleEndian>()
        }
        .map_err(|e| IoError::BinaryReadError {
            reason: e.to_string(),
        })?;

        data.push(value as f64);
    }

    Ok(Array1::from_vec(data))
}
