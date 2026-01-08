//! Bruker 1D spectrum reader.
//!
//! Reads processed 1D spectra from pdata/N/1r files or raw FID from fid files.

use std::fs::File;
use std::io::{Read, BufReader};
use std::path::Path;

use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use ndarray::Array1;
use uuid::Uuid;

use super::{BrukerAcqParams, BrukerProcParams, BrukerImportOptions};
use crate::data::spectrum::{Spectrum1D, SpectrumMetadata};
use crate::io::error::{IoError, IoResult};

/// Import a Bruker 1D spectrum.
///
/// If `options.processed` is true, reads from pdata/N/1r.
/// Otherwise, reads the FID from the fid file.
pub fn import_bruker_1d(options: &BrukerImportOptions) -> IoResult<Spectrum1D> {
    let exp_dir = Path::new(&options.path);

    if !exp_dir.exists() {
        return Err(IoError::FileNotFound {
            path: options.path.clone(),
        });
    }

    // Load acquisition parameters
    let acq_params = BrukerAcqParams::load(exp_dir)?;

    if options.processed {
        import_processed_1d(exp_dir, &acq_params, options)
    } else {
        import_fid_1d(exp_dir, &acq_params, options)
    }
}

/// Import processed 1D spectrum from pdata directory.
fn import_processed_1d(
    exp_dir: &Path,
    acq_params: &BrukerAcqParams,
    options: &BrukerImportOptions,
) -> IoResult<Spectrum1D> {
    let pdata_dir = exp_dir.join("pdata").join(options.procno.to_string());

    if !pdata_dir.exists() {
        return Err(IoError::InvalidBrukerStructure {
            reason: format!("pdata/{} directory not found", options.procno),
        });
    }

    // Load processing parameters
    let proc_params = BrukerProcParams::load(&pdata_dir)?;

    // Read the 1r file (real data)
    let data_path = pdata_dir.join("1r");
    let real_data = read_bruker_binary_1d(&data_path, proc_params.bytordp, proc_params.nc_proc)?;

    // Optionally read imaginary part (1i)
    let imag_data = {
        let imag_path = pdata_dir.join("1i");
        if imag_path.exists() {
            read_bruker_binary_1d(&imag_path, proc_params.bytordp, proc_params.nc_proc).ok()
        } else {
            None
        }
    };

    // Build metadata
    let experiment_type = acq_params.infer_experiment_type();
    let nucleus = BrukerAcqParams::parse_nucleus(&acq_params.nuc1);

    let name = options
        .name
        .clone()
        .or_else(|| acq_params.title.clone())
        .unwrap_or_else(|| format!("{} 1D", acq_params.nuc1));

    let n_points = real_data.len();

    let metadata = SpectrumMetadata {
        id: Uuid::new_v4(),
        name,
        experiment_type,
        nucleus_types: vec![nucleus],
        spectral_width_hz: vec![proc_params.sw_p * proc_params.sf],
        carrier_offset_ppm: vec![proc_params.offset - proc_params.sw_p / 2.0],
        spectrometer_frequency_mhz: proc_params.sf,
        original_points: vec![acq_params.td / 2],
        processed_points: vec![proc_params.si],
    };

    // Calculate PPM axis
    // Bruker: OFFSET is the ppm of the leftmost point (highest ppm)
    // PPM decreases from left to right
    let ppm_axis = Array1::linspace(
        proc_params.offset,
        proc_params.offset - proc_params.sw_p,
        n_points,
    );

    Ok(Spectrum1D {
        metadata,
        real: real_data,
        imag: imag_data,
        ppm_axis,
        is_processed: true,
    })
}

/// Import raw FID from fid file.
fn import_fid_1d(
    exp_dir: &Path,
    acq_params: &BrukerAcqParams,
    options: &BrukerImportOptions,
) -> IoResult<Spectrum1D> {
    let fid_path = exp_dir.join("fid");

    if !fid_path.exists() {
        return Err(IoError::FileNotFound {
            path: fid_path.display().to_string(),
        });
    }

    // Read interleaved real/imag FID data
    let (real_data, imag_data) = read_bruker_fid(&fid_path, acq_params)?;

    // Build metadata
    let experiment_type = acq_params.infer_experiment_type();
    let nucleus = BrukerAcqParams::parse_nucleus(&acq_params.nuc1);

    let name = options
        .name
        .clone()
        .or_else(|| acq_params.title.clone())
        .unwrap_or_else(|| format!("{} FID", acq_params.nuc1));

    let n_points = real_data.len();

    // Calculate carrier offset in ppm
    let carrier_ppm = acq_params.o1 / acq_params.bf1;

    let metadata = SpectrumMetadata {
        id: Uuid::new_v4(),
        name,
        experiment_type,
        nucleus_types: vec![nucleus],
        spectral_width_hz: vec![acq_params.sw_h],
        carrier_offset_ppm: vec![carrier_ppm],
        spectrometer_frequency_mhz: acq_params.bf1,
        original_points: vec![n_points],
        processed_points: vec![n_points],
    };

    // PPM axis for FID (will be recalculated after FFT)
    let sw_ppm = acq_params.sw_h / acq_params.bf1;
    let ppm_axis = Array1::linspace(
        carrier_ppm + sw_ppm / 2.0,
        carrier_ppm - sw_ppm / 2.0,
        n_points,
    );

    Ok(Spectrum1D {
        metadata,
        real: real_data,
        imag: Some(imag_data),
        ppm_axis,
        is_processed: false,
    })
}

/// Read Bruker binary 1D processed data (1r or 1i file).
fn read_bruker_binary_1d(path: &Path, bytordp: u8, nc_proc: i32) -> IoResult<Array1<f64>> {
    let file = File::open(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            IoError::FileNotFound {
                path: path.display().to_string(),
            }
        } else {
            IoError::Io(e)
        }
    })?;

    let file_size = file.metadata()?.len() as usize;
    let n_points = file_size / 4; // int32 = 4 bytes

    let mut reader = BufReader::new(file);
    let mut data = Vec::with_capacity(n_points);

    // NC_proc is a scaling factor: data = raw_data * 2^NC_proc
    let scale = 2.0_f64.powi(nc_proc);

    for _ in 0..n_points {
        let value = if bytordp == 0 {
            reader.read_i32::<LittleEndian>()
        } else {
            reader.read_i32::<BigEndian>()
        }
        .map_err(|e| IoError::BinaryReadError {
            reason: e.to_string(),
        })?;

        data.push(value as f64 * scale);
    }

    Ok(Array1::from_vec(data))
}

/// Read Bruker FID file (interleaved real/imag int32).
fn read_bruker_fid(path: &Path, params: &BrukerAcqParams) -> IoResult<(Array1<f64>, Array1<f64>)> {
    let file = File::open(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            IoError::FileNotFound {
                path: path.display().to_string(),
            }
        } else {
            IoError::Io(e)
        }
    })?;

    // TD is the total number of data points (real + imag interleaved)
    let n_complex = params.td / 2;

    let mut reader = BufReader::new(file);
    let mut real_data = Vec::with_capacity(n_complex);
    let mut imag_data = Vec::with_capacity(n_complex);

    let is_big_endian = params.bytorda != 0;

    for _ in 0..n_complex {
        let (re, im) = if params.dtypa == 2 {
            // Float64 data
            let re = if is_big_endian {
                reader.read_f64::<BigEndian>()
            } else {
                reader.read_f64::<LittleEndian>()
            }
            .map_err(|e| IoError::BinaryReadError {
                reason: e.to_string(),
            })?;

            let im = if is_big_endian {
                reader.read_f64::<BigEndian>()
            } else {
                reader.read_f64::<LittleEndian>()
            }
            .map_err(|e| IoError::BinaryReadError {
                reason: e.to_string(),
            })?;

            (re, im)
        } else {
            // Int32 data
            let re = if is_big_endian {
                reader.read_i32::<BigEndian>()
            } else {
                reader.read_i32::<LittleEndian>()
            }
            .map_err(|e| IoError::BinaryReadError {
                reason: e.to_string(),
            })? as f64;

            let im = if is_big_endian {
                reader.read_i32::<BigEndian>()
            } else {
                reader.read_i32::<LittleEndian>()
            }
            .map_err(|e| IoError::BinaryReadError {
                reason: e.to_string(),
            })? as f64;

            (re, im)
        };

        real_data.push(re);
        imag_data.push(im);
    }

    Ok((Array1::from_vec(real_data), Array1::from_vec(imag_data)))
}
