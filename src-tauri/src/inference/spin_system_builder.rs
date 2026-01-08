//! Spin system builder from unlabeled peak lists.
//!
//! **Philosophy**: All spectra are equally important. We build spin systems from
//! ALL available data simultaneously, then merge based on TOCSY correlations.
//!
//! Algorithm:
//! 1. Create proto-spin-systems from each data source:
//!    - 15N-HSQC: (N, H_backbone)
//!    - 13C-HSQC: (C, H_attached) for each C-H pair
//! 2. Build TOCSY correlation graph linking related protons
//! 3. Merge proto-spin-systems that share TOCSY-correlated protons
//! 4. Classify atoms using BMRB statistics

use std::collections::{HashMap, HashSet};
use uuid::Uuid;

use crate::data::{PeakExperimentType, ShiftCandidate, SpinSystem, UnlabeledPeak};
use crate::testdata::BMRBDatabase;

/// Parameters for spin system building.
#[derive(Debug, Clone)]
pub struct SpinSystemBuildParams {
    /// Tolerance for matching proton positions (ppm)
    pub h_tolerance: f64,
    /// Tolerance for matching nitrogen positions (ppm)
    pub n_tolerance: f64,
    /// Tolerance for matching carbon positions (ppm)
    pub c_tolerance: f64,
    /// Minimum TOCSY intensity relative to max (0.0-1.0)
    pub min_tocsy_intensity: f64,
}

impl Default for SpinSystemBuildParams {
    fn default() -> Self {
        Self {
            h_tolerance: 0.03,     // 0.03 ppm for protons
            n_tolerance: 0.3,      // 0.3 ppm for nitrogen
            c_tolerance: 0.3,      // 0.3 ppm for carbon
            min_tocsy_intensity: 0.05, // 5% of max intensity
        }
    }
}

/// A proto-spin-system: partial information from a single source.
#[derive(Debug, Clone)]
struct ProtoSpinSystem {
    id: Uuid,
    /// Proton shifts with their source (for merging)
    protons: HashMap<String, f64>, // key = "H", "HA", "HB", etc. or "H_unknown"
    /// Carbon shifts
    carbons: HashMap<String, (f64, f64)>, // key = "CA", "CB", etc., value = (shift, score)
    /// Nitrogen shift (only from 15N-HSQC)
    n_shift: Option<f64>,
    /// Source peak ID
    source_peak_id: Uuid,
}

/// Build spin systems from unlabeled peak lists.
///
/// All spectra contribute equally to spin system building.
pub fn build_spin_systems(
    hsqc_15n_peaks: &[UnlabeledPeak],
    hsqc_13c_peaks: &[UnlabeledPeak],
    tocsy_peaks: &[UnlabeledPeak],
    params: &SpinSystemBuildParams,
) -> Vec<SpinSystem> {
    let bmrb = BMRBDatabase::load_embedded();
    build_spin_systems_with_bmrb(hsqc_15n_peaks, hsqc_13c_peaks, tocsy_peaks, params, &bmrb)
}

/// Build spin systems treating ALL spectra equally.
///
/// Algorithm:
/// 1. Create proto-spin-systems from 15N-HSQC (N, H) and 13C-HSQC (C, H)
/// 2. Build TOCSY correlation graph
/// 3. Merge proto-spin-systems that share TOCSY-correlated protons
/// 4. Classify atoms using BMRB
pub fn build_spin_systems_with_bmrb(
    hsqc_15n_peaks: &[UnlabeledPeak],
    hsqc_13c_peaks: &[UnlabeledPeak],
    tocsy_peaks: &[UnlabeledPeak],
    params: &SpinSystemBuildParams,
    bmrb: &BMRBDatabase,
) -> Vec<SpinSystem> {
    // Step 1: Create proto-spin-systems from all HSQC peaks
    let mut protos: Vec<ProtoSpinSystem> = Vec::new();

    // From 15N-HSQC: each peak gives (N, H_backbone)
    for peak in hsqc_15n_peaks {
        if peak.experiment_type != PeakExperimentType::Hsqc15N {
            continue;
        }
        let mut proto = ProtoSpinSystem {
            id: Uuid::new_v4(),
            protons: HashMap::new(),
            carbons: HashMap::new(),
            n_shift: Some(peak.position_ppm[0]),
            source_peak_id: peak.id,
        };
        proto.protons.insert("H".to_string(), peak.position_ppm[1]);
        protos.push(proto);
    }

    // From 13C-HSQC: each peak gives (C, H_attached)
    for peak in hsqc_13c_peaks {
        if peak.experiment_type != PeakExperimentType::Hsqc13C {
            continue;
        }
        let c_shift = peak.position_ppm[0];
        let h_shift = peak.position_ppm[1];

        // Classify C-H pair using BMRB
        if let Some((c_name, h_name)) = classify_ch_pair_by_carbon(c_shift, h_shift, bmrb) {
            let mut proto = ProtoSpinSystem {
                id: Uuid::new_v4(),
                protons: HashMap::new(),
                carbons: HashMap::new(),
                n_shift: None,
                source_peak_id: peak.id,
            };
            proto.protons.insert(h_name, h_shift);
            proto.carbons.insert(c_name, (c_shift, 1.0));
            protos.push(proto);
        }
    }

    tracing::debug!("Created {} proto-spin-systems from HSQC peaks", protos.len());

    // Step 2: Build TOCSY correlation graph
    let tocsy_graph = build_tocsy_graph(tocsy_peaks, params);

    // Step 3: Compute soft affinities between all protos
    let affinity = compute_proto_affinities(&protos, &tocsy_graph, params);

    // Step 4: Soft clustering with affinity threshold
    let min_affinity = 0.1; // Allow weak correlations to be considered
    let clusters = cluster_protos_soft(&protos, &affinity, min_affinity);

    tracing::debug!("Clustered into {} spin system groups", clusters.len());

    // Step 5: Convert clusters to SpinSystems with soft weights
    let mut spin_systems = Vec::new();
    for cluster in clusters {
        if let Some(ss) = build_spin_system_from_cluster(&cluster, &protos, bmrb) {
            spin_systems.push(ss);
        }
    }

    tracing::info!(
        "Built {} spin systems from {} 15N-HSQC + {} 13C-HSQC peaks",
        spin_systems.len(),
        hsqc_15n_peaks.len(),
        hsqc_13c_peaks.len()
    );

    spin_systems
}

/// Build TOCSY correlation graph: which proton shifts are correlated?
/// Returns map of proton shift -> set of correlated proton shifts.
fn build_tocsy_graph(
    tocsy_peaks: &[UnlabeledPeak],
    params: &SpinSystemBuildParams,
) -> HashMap<i32, HashSet<i32>> {
    // Discretize shifts to integers (0.01 ppm bins) for easy lookup
    let discretize = |shift: f64| (shift * 100.0).round() as i32;

    let mut graph: HashMap<i32, HashSet<i32>> = HashMap::new();

    let max_intensity = tocsy_peaks.iter().map(|p| p.intensity).fold(0.0f64, f64::max);
    if max_intensity == 0.0 {
        return graph;
    }

    for peak in tocsy_peaks {
        if peak.experiment_type != PeakExperimentType::Tocsy {
            continue;
        }

        let rel_intensity = peak.intensity / max_intensity;
        if rel_intensity < params.min_tocsy_intensity {
            continue;
        }

        let h1 = discretize(peak.position_ppm[0]);
        let h2 = discretize(peak.position_ppm[1]);

        // Skip diagonal (same proton)
        if (h1 - h2).abs() <= 3 { // Within 0.03 ppm
            continue;
        }

        // Add bidirectional correlation
        graph.entry(h1).or_default().insert(h2);
        graph.entry(h2).or_default().insert(h1);
    }

    graph
}

/// Compute soft correlation scores between all proto-spin-systems.
/// Returns affinity matrix where affinity[i][j] is how strongly protos i and j are linked.
fn compute_proto_affinities(
    protos: &[ProtoSpinSystem],
    tocsy_graph: &HashMap<i32, HashSet<i32>>,
    params: &SpinSystemBuildParams,
) -> Vec<Vec<f64>> {
    let n = protos.len();
    let mut affinity = vec![vec![0.0; n]; n];
    let discretize = |shift: f64| (shift * 100.0).round() as i32;
    let sigma = params.h_tolerance * 100.0; // In discretized units

    for i in 0..n {
        affinity[i][i] = 1.0; // Self-affinity
        for j in (i + 1)..n {
            let score = compute_pair_affinity(&protos[i], &protos[j], tocsy_graph, &discretize, sigma);
            affinity[i][j] = score;
            affinity[j][i] = score;
        }
    }

    affinity
}

/// Compute soft affinity between two protos based on TOCSY correlations.
fn compute_pair_affinity(
    p1: &ProtoSpinSystem,
    p2: &ProtoSpinSystem,
    tocsy_graph: &HashMap<i32, HashSet<i32>>,
    discretize: &impl Fn(f64) -> i32,
    sigma: f64,
) -> f64 {
    let mut max_score = 0.0f64;

    for (name1, &h1) in &p1.protons {
        let h1_bin = discretize(h1);

        for (name2, &h2) in &p2.protons {
            let h2_bin = discretize(h2);

            // Same backbone H from 15N-HSQC: strong affinity
            if name1 == "H" && name2 == "H" {
                let delta = (h1_bin - h2_bin).abs() as f64;
                let score = (-0.5 * (delta / sigma).powi(2)).exp();
                max_score = max_score.max(score);
            }

            // TOCSY correlation: Gaussian-weighted affinity
            if let Some(correlations) = tocsy_graph.get(&h1_bin) {
                for &corr_bin in correlations {
                    let delta = (corr_bin - h2_bin).abs() as f64;
                    let score = (-0.5 * (delta / sigma).powi(2)).exp();
                    max_score = max_score.max(score);
                }
            }
        }
    }

    max_score
}

/// Group protos into spin systems using soft clustering.
/// Each spin system is seeded by a 15N-HSQC proto (has backbone H).
/// 13C-HSQC protos are assigned with soft weights to the best-matching spin system.
fn cluster_protos_soft(
    protos: &[ProtoSpinSystem],
    affinity: &[Vec<f64>],
    min_affinity: f64,
) -> Vec<Vec<(usize, f64)>> {
    // Seeds: protos with backbone H (from 15N-HSQC)
    let seeds: Vec<usize> = protos
        .iter()
        .enumerate()
        .filter(|(_, p)| p.protons.contains_key("H"))
        .map(|(i, _)| i)
        .collect();

    // Also add 13C-HSQC protos that have no affinity to any seed as their own clusters
    // These are "orphan" spin systems (like N-terminal residues)
    let mut orphans: Vec<usize> = Vec::new();
    for (i, proto) in protos.iter().enumerate() {
        if proto.protons.contains_key("H") {
            continue; // Already a seed
        }
        let max_seed_affinity = seeds.iter().map(|&s| affinity[i][s]).fold(0.0f64, f64::max);
        if max_seed_affinity < min_affinity {
            orphans.push(i);
        }
    }

    // Group orphans that are strongly affiliated with each other
    let mut orphan_groups: Vec<Vec<usize>> = Vec::new();
    let mut assigned_orphans: HashSet<usize> = HashSet::new();

    for &orphan in &orphans {
        if assigned_orphans.contains(&orphan) {
            continue;
        }
        let mut group = vec![orphan];
        assigned_orphans.insert(orphan);

        for &other in &orphans {
            if assigned_orphans.contains(&other) {
                continue;
            }
            if affinity[orphan][other] >= min_affinity {
                group.push(other);
                assigned_orphans.insert(other);
            }
        }
        orphan_groups.push(group);
    }

    // Build clusters: each seed becomes a cluster, plus orphan groups
    let mut clusters: Vec<Vec<(usize, f64)>> = Vec::new();

    // Seed clusters
    for &seed in &seeds {
        let mut cluster = vec![(seed, 1.0)];

        // Add affiliated protos with their affinity scores
        for (i, proto) in protos.iter().enumerate() {
            if i == seed || proto.protons.contains_key("H") {
                continue;
            }
            let aff = affinity[i][seed];
            if aff >= min_affinity {
                cluster.push((i, aff));
            }
        }

        clusters.push(cluster);
    }

    // Orphan clusters
    for group in orphan_groups {
        let cluster: Vec<(usize, f64)> = group.into_iter().map(|i| (i, 1.0)).collect();
        clusters.push(cluster);
    }

    clusters
}

/// Build a SpinSystem from a soft cluster of proto-spin-systems with affinity weights.
fn build_spin_system_from_cluster(
    cluster: &[(usize, f64)], // (proto_index, affinity_weight)
    protos: &[ProtoSpinSystem],
    _bmrb: &BMRBDatabase,
) -> Option<SpinSystem> {
    // Collect all data from the cluster, weighted by affinity
    let mut all_protons: HashMap<String, Vec<(f64, f64)>> = HashMap::new(); // (shift, weight)
    let mut all_carbons: HashMap<String, Vec<(f64, f64)>> = HashMap::new();
    let mut n_shifts: Vec<(f64, f64)> = Vec::new();
    let mut anchor_peak_id: Option<Uuid> = None;

    for &(idx, weight) in cluster {
        let proto = &protos[idx];

        for (name, &shift) in &proto.protons {
            all_protons.entry(name.clone()).or_default().push((shift, weight));
        }

        for (name, &(shift, score)) in &proto.carbons {
            // Combine original score with affinity weight
            all_carbons.entry(name.clone()).or_default().push((shift, score * weight));
        }

        if let Some(n) = proto.n_shift {
            n_shifts.push((n, weight));
            if anchor_peak_id.is_none() {
                anchor_peak_id = Some(proto.source_peak_id);
            }
        }
    }

    // Determine anchor: prefer backbone H from 15N-HSQC, fall back to HA
    let (anchor_h, has_backbone_n) = if let Some(h_shifts) = all_protons.get("H") {
        // Weighted average of backbone H
        let total_weight: f64 = h_shifts.iter().map(|(_, w)| w).sum();
        let weighted_h: f64 = h_shifts.iter().map(|(h, w)| h * w).sum::<f64>() / total_weight.max(0.001);
        (weighted_h, true)
    } else if let Some(ha_shifts) = all_protons.get("HA") {
        // N-terminal: use weighted average of HA
        let total_weight: f64 = ha_shifts.iter().map(|(_, w)| w).sum();
        let weighted_ha: f64 = ha_shifts.iter().map(|(h, w)| h * w).sum::<f64>() / total_weight.max(0.001);
        (weighted_ha, false)
    } else {
        return None;
    };

    // Weighted average N shift
    let n_shift = if n_shifts.is_empty() {
        0.0
    } else {
        let total_weight: f64 = n_shifts.iter().map(|(_, w)| w).sum();
        n_shifts.iter().map(|(n, w)| n * w).sum::<f64>() / total_weight.max(0.001)
    };

    // Create spin system
    let anchor = if has_backbone_n && n_shift > 0.0 {
        UnlabeledPeak::hsqc_15n(n_shift, anchor_h, 1.0)
    } else {
        UnlabeledPeak::hsqc_15n(0.0, anchor_h, 1.0)
    };

    let mut ss = SpinSystem::new(&anchor);
    if let Some(id) = anchor_peak_id {
        ss.anchor_peak_id = id;
    }
    ss.has_backbone_nh = has_backbone_n && n_shift > 0.0;

    // Add weighted proton shifts
    for (name, shifts) in &all_protons {
        if name == "H" && has_backbone_n {
            continue;
        }
        let total_weight: f64 = shifts.iter().map(|(_, w)| w).sum();
        let weighted_avg: f64 = shifts.iter().map(|(h, w)| h * w).sum::<f64>() / total_weight.max(0.001);
        ss.add_proton_shift(name, weighted_avg);
    }

    // Add carbon candidates with combined scores
    for (name, shifts_scores) in &all_carbons {
        for &(shift, score) in shifts_scores {
            let candidate = ShiftCandidate::new(shift, score);
            ss.add_carbon_candidate(name, candidate);
        }
    }

    ss.finalize_carbon_assignments();

    Some(ss)
}

/// Find TOCSY correlations from a given proton position with **soft matching**.
///
/// Returns tuples of (correlated_h_shift, intensity, match_score).
/// The match_score indicates how well the source matches the TOCSY peak (0-1).
/// Spin systems with backbone H closer to the TOCSY peak get higher scores.
fn find_tocsy_correlations(
    source_h: f64,
    tocsy_peaks: &[UnlabeledPeak],
    h_tolerance: f64,
    min_relative_intensity: f64,
) -> Vec<(f64, f64)> {
    // Find max intensity for normalization
    let max_intensity = tocsy_peaks
        .iter()
        .map(|p| p.intensity)
        .fold(0.0f64, f64::max);

    if max_intensity == 0.0 {
        return Vec::new();
    }

    let mut correlations = Vec::new();

    for peak in tocsy_peaks {
        if peak.experiment_type != PeakExperimentType::Tocsy {
            continue;
        }

        let h1 = peak.position_ppm[0];
        let h2 = peak.position_ppm[1];
        let rel_intensity = peak.intensity / max_intensity;

        if rel_intensity < min_relative_intensity {
            continue;
        }

        // Use Gaussian soft matching instead of hard tolerance
        let sigma = h_tolerance;

        // TOCSY is symmetric - check both dimensions with soft matching
        let delta1 = (h1 - source_h).abs();
        let delta2 = (h2 - source_h).abs();

        let score1 = (-0.5 * (delta1 / sigma).powi(2)).exp();
        let score2 = (-0.5 * (delta2 / sigma).powi(2)).exp();

        // Require minimum match score of 0.1 (roughly 2 sigma)
        let min_score = 0.1;

        if score1 > min_score && score2 < 0.5 {
            // h1 matches source, h2 is the correlation
            // Weight the correlation by match quality
            correlations.push((h2, peak.intensity * score1));
        } else if score2 > min_score && score1 < 0.5 {
            // h2 matches source, h1 is the correlation
            correlations.push((h1, peak.intensity * score2));
        }
    }

    // Sort by intensity (strongest first)
    correlations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    correlations
}

/// Classify a proton chemical shift to an atom name using BMRB statistics.
///
/// Returns None for the backbone amide H itself (already captured).
/// Uses z-score matching against BMRB database for all proton types.
fn classify_proton_shift_bmrb(h_shift: f64, backbone_h: f64, bmrb: &BMRBDatabase) -> Option<String> {
    // Skip if it's the backbone amide H (within tolerance)
    if (h_shift - backbone_h).abs() < 0.1 {
        return None;
    }

    // Skip if it's in the backbone amide region (7.5-9.5 ppm)
    if (7.5..=9.5).contains(&h_shift) {
        return None;
    }

    // Get all proton atoms from BMRB
    let all_protons = bmrb.get_nucleus_atoms("H");

    // Aggregate statistics by atom type (HA, HB, HG, etc.)
    // We need to find which atom TYPE best matches this shift
    let atom_types = ["HA", "HA2", "HA3", "HB", "HB2", "HB3", "HG", "HG2", "HG3",
                      "HG12", "HG13", "HD", "HD2", "HD3", "HD21", "HD22",
                      "HE", "HE2", "HE3", "HE21", "HE22", "HZ"];

    let mut best_match: Option<(String, f64)> = None;

    for atom_type in atom_types {
        // Collect all BMRB entries for this atom type across all residues
        let matching_stats: Vec<_> = all_protons
            .iter()
            .filter(|s| s.atom_name == atom_type)
            .collect();

        if matching_stats.is_empty() {
            continue;
        }

        // Calculate average mean and std_dev across all residues for this atom type
        let total_mean: f64 = matching_stats.iter().map(|s| s.mean).sum::<f64>() / matching_stats.len() as f64;
        let avg_std: f64 = matching_stats.iter().map(|s| s.std_dev).sum::<f64>() / matching_stats.len() as f64;

        // Calculate z-score (how many std deviations from the mean)
        let std_dev = avg_std.max(0.1); // Avoid division by zero
        let z_score = ((h_shift - total_mean) / std_dev).abs();

        // Keep the best match (lowest z-score)
        if z_score < 3.0 {
            // Only consider matches within 3 sigma
            if best_match.is_none() || z_score < best_match.as_ref().unwrap().1 {
                // Normalize atom name (HA2/HA3 -> HA, HB2/HB3 -> HB, etc.)
                let normalized = normalize_proton_name(atom_type);
                best_match = Some((normalized, z_score));
            }
        }
    }

    best_match.map(|(name, _)| name)
}

/// Normalize proton atom names to canonical form.
/// e.g., HA2, HA3 -> HA; HB2, HB3 -> HB; etc.
fn normalize_proton_name(atom_name: &str) -> String {
    match atom_name {
        "HA2" | "HA3" => "HA".to_string(),
        "HB2" | "HB3" => "HB".to_string(),
        "HG2" | "HG3" | "HG12" | "HG13" => "HG".to_string(),
        "HD2" | "HD3" | "HD21" | "HD22" => "HD".to_string(),
        "HE2" | "HE3" | "HE21" | "HE22" => "HE".to_string(),
        _ => atom_name.to_string(),
    }
}

/// Classify a proton chemical shift to an atom name based on typical ranges (fallback).
///
/// Returns None for the backbone amide H itself (already captured).
#[allow(dead_code)]
fn classify_proton_shift(h_shift: f64, backbone_h: f64) -> Option<String> {
    // Skip if it's the backbone amide H (within tolerance)
    if (h_shift - backbone_h).abs() < 0.1 {
        return None;
    }

    // Chemical shift ranges for proton classification
    // These are approximate - actual assignment needs BMRB matching
    match h_shift {
        h if (3.5..=5.5).contains(&h) => {
            // Alpha proton region
            Some("HA".to_string())
        }
        h if (0.5..=3.5).contains(&h) => {
            // Aliphatic sidechain region
            // Finer classification based on shift
            if h < 1.2 {
                Some("HB".to_string()) // Could be methyl HB (ALA, VAL, etc.)
            } else if h < 2.0 {
                Some("HG".to_string()) // Often HG region
            } else if h < 2.8 {
                Some("HB".to_string()) // Methylene HB region
            } else {
                Some("HD".to_string()) // Could be various
            }
        }
        h if (6.0..=8.0).contains(&h) => {
            // Aromatic region (not backbone amide)
            Some("HE".to_string()) // Aromatic protons
        }
        _ => None,
    }
}

/// Classify a 13C-HSQC peak to carbon and attached proton names.
///
/// Returns (carbon_name, proton_name) based on chemical shift ranges.
/// Uses strict ranges to avoid misclassification.
fn classify_carbon_hsqc_peak(c_shift: f64, h_shift: f64) -> Option<(String, String)> {
    // Carbon chemical shift classification with stricter ranges
    // Key insight: CA/HA has distinct shift ranges from CB/HB

    // CA region: Carbon 44-66 ppm, Proton 3.5-5.5 ppm (alpha proton region)
    // GLY CA is ~45 ppm with HA ~3.9-4.0 ppm
    // Most others CA ~50-65 ppm with HA ~4.0-5.0 ppm
    if (44.0..=66.0).contains(&c_shift) && (3.5..=5.5).contains(&h_shift) {
        return Some(("CA".to_string(), "HA".to_string()));
    }

    // CB methyl region: Carbon 15-25 ppm, Proton 0.5-2.0 ppm
    // ALA CB ~19 ppm, VAL/ILE methyls ~17-22 ppm
    if (15.0..=25.0).contains(&c_shift) && (0.5..=2.0).contains(&h_shift) {
        return Some(("CB".to_string(), "HB".to_string()));
    }

    // CB methylene region: Carbon 25-45 ppm, Proton 1.5-3.5 ppm
    // Most CB are ~28-42 ppm with HB ~2-3 ppm
    if (25.0..=45.0).contains(&c_shift) && (1.5..=3.5).contains(&h_shift) {
        return Some(("CB".to_string(), "HB".to_string()));
    }

    // CB hydroxyl (SER/THR): Carbon 60-72 ppm, Proton 3.7-4.2 ppm
    // SER CB ~63 ppm with HB ~3.9 ppm
    // THR CB ~69 ppm with HB ~4.2 ppm
    // Note: This overlaps with CA range, but the proton shift is lower
    if (60.0..=72.0).contains(&c_shift) && (3.7..=4.5).contains(&h_shift) {
        // Distinguish from CA: SER/THR HB is typically > 3.7 ppm
        // CA protons (HA) for these residues are typically > 4.3 ppm
        if h_shift < 4.3 {
            return Some(("CB".to_string(), "HB".to_string()));
        }
    }

    // CG region: Carbon 20-35 ppm, Proton 1.0-2.5 ppm
    if (20.0..=35.0).contains(&c_shift) && (1.0..=2.5).contains(&h_shift) {
        // Could be CG for LEU, ILE, etc.
        return Some(("CG".to_string(), "HG".to_string()));
    }

    // CD region: Carbon 25-45 ppm, Proton 1.5-3.5 ppm
    if (25.0..=50.0).contains(&c_shift) && (2.5..=4.0).contains(&h_shift) {
        return Some(("CD".to_string(), "HD".to_string()));
    }

    // Aromatic region: Carbon 110-145 ppm, Proton 6.5-8.0 ppm
    if (110.0..=145.0).contains(&c_shift) && (6.5..=8.0).contains(&h_shift) {
        return Some(("CD".to_string(), "HD".to_string()));
    }

    None
}

/// Assign C-H pairs from 13C-HSQC using carbon shifts as the PRIMARY discriminator.
///
/// Key insight: Carbon shifts are MUCH more discriminating than proton shifts:
/// - CA: ~45-65 ppm (GLY ~45, others ~50-65)
/// - CB aliphatic: ~15-45 ppm
/// - CB hydroxyl (SER/THR): ~60-70 ppm
/// - Aromatic C: ~110-145 ppm
///
/// When we see C=58 ppm with H=4.67 ppm, the carbon tells us this is CA,
/// so the proton must be HA (not aromatic HE which would have C~120 ppm).
fn assign_ch_pairs_from_hsqc(
    ss: &mut SpinSystem,
    hsqc_13c_peaks: &[UnlabeledPeak],
    tocsy_correlations: &[(f64, f64)],
    params: &SpinSystemBuildParams,
    bmrb: &BMRBDatabase,
) {
    // Collect all TOCSY-correlated proton positions (these belong to this spin system)
    let mut tocsy_protons: Vec<f64> = tocsy_correlations.iter().map(|(h, _)| *h).collect();

    // Remove duplicates within tolerance
    tocsy_protons.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tocsy_protons.dedup_by(|a, b| (*a - *b).abs() < 0.02);

    // Match 13C-HSQC peaks to TOCSY-correlated protons using soft matching
    for peak in hsqc_13c_peaks {
        if peak.experiment_type != PeakExperimentType::Hsqc13C {
            continue;
        }

        let c_shift = peak.position_ppm[0];
        let h_shift = peak.position_ppm[1];

        // Find best matching TOCSY proton and compute Gaussian match score
        let mut best_match: Option<(f64, f64)> = None; // (proton_shift, score)

        for &tocsy_h in &tocsy_protons {
            let delta = (h_shift - tocsy_h).abs();

            // Use HARD threshold: proton must match within 0.005 ppm
            // This prevents cross-contamination from nearly-overlapping protons
            // (e.g., ILE HG21=0.872 vs LEU HD11=0.870 differ by only 0.002 ppm)
            // For synthetic "perfect" data, HSQC and TOCSY protons from same
            // residue will be EXACTLY equal (same KDE mode).
            let hard_tolerance = 0.005; // 5 Hz at 1000 MHz, very tight
            if delta < hard_tolerance {
                let match_score = 1.0 - (delta / hard_tolerance); // Linear falloff
                if best_match.is_none() || match_score > best_match.unwrap().1 {
                    best_match = Some((tocsy_h, match_score));
                }
            }
        }

        // If we found a matching TOCSY proton, classify the C-H pair
        if let Some((matched_h, match_score)) = best_match {
            // Use CARBON shift as PRIMARY classifier (much more discriminating!)
            if let Some((carbon_name, proton_name)) = classify_ch_pair_by_carbon(c_shift, h_shift, bmrb) {
                // Add carbon as candidate with match score
                let candidate = ShiftCandidate::with_peak(c_shift, match_score, peak.id);
                ss.add_carbon_candidate(&carbon_name, candidate);

                // Add the proton with the name derived from the carbon classification
                if !ss.proton_shifts.contains_key(&proton_name) {
                    ss.add_proton_shift(&proton_name, matched_h);
                }
            }
        }
    }

    // Finalize: pick best candidates for carbon_shifts
    ss.finalize_carbon_assignments();
}

/// Classify a C-H pair using BMRB statistics with carbon as PRIMARY discriminator.
///
/// Key insight: Don't average across residues! Check each residue's specific statistics.
/// SER CB (~63 ppm) is VERY different from ALA CB (~19 ppm).
/// Find the best (residue, atom) match, then return just the atom type.
fn classify_ch_pair_by_carbon(c_shift: f64, h_shift: f64, bmrb: &BMRBDatabase) -> Option<(String, String)> {
    // Carbon-proton pairs we care about (excludes CO carbons which aren't in HSQC)
    // Includes both aliphatic and aromatic carbons
    let ch_pairs = [
        ("CA", "HA"),
        ("CB", "HB"),
        ("CG", "HG"),
        ("CG1", "HG"),
        ("CG2", "HG"),
        ("CD", "HD"),
        ("CD1", "HD"),
        ("CD2", "HD"),
        ("CE", "HE"),
        ("CE1", "HE"),
        ("CE2", "HE"),
        ("CE3", "HE3"),   // TRP aromatic
        ("CZ", "HZ"),
        ("CZ2", "HZ2"),   // TRP aromatic
        ("CZ3", "HZ3"),   // TRP aromatic
        ("CH2", "HH2"),   // TRP aromatic
    ];

    let mut best_match: Option<(&str, &str, f64)> = None; // (carbon, proton, z_score)

    // Check EACH residue type separately - don't average!
    for residue in bmrb.residue_names() {
        for (carbon_type, proton_type) in ch_pairs {
            // Get this specific residue's carbon statistics
            if let Some(stats) = bmrb.get(&residue, carbon_type) {
                let std_dev = stats.std_dev.max(1.0);
                let z_score = ((c_shift - stats.mean) / std_dev).abs();

                // Only consider if within 2.5 sigma (tighter threshold)
                if z_score < 2.5 {
                    if best_match.is_none() || z_score < best_match.unwrap().2 {
                        best_match = Some((carbon_type, proton_type, z_score));
                    }
                }
            }
        }
    }

    best_match.map(|(c, h, _)| {
        // Normalize names (CG1/CG2 -> CG, etc.)
        let c_norm = normalize_carbon_name(c);
        let h_norm = h.to_string();
        (c_norm, h_norm)
    })
}

/// Classify a 13C-HSQC peak (carbon and proton) using BMRB statistics.
///
/// Returns (carbon_name, proton_name) for the best matching C-H pair.
fn classify_carbon_shift_bmrb(c_shift: f64, h_shift: f64, bmrb: &BMRBDatabase) -> Option<(String, String)> {
    // Get all carbon atoms from BMRB
    let all_carbons = bmrb.get_nucleus_atoms("C");

    // Carbon atom types to consider (excluding CO carbons which aren't in 13C-HSQC)
    // Includes both aliphatic and aromatic carbons (TRP CE3, CZ2, CZ3, CH2)
    let carbon_types = ["CA", "CB", "CG", "CG1", "CG2", "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CE3", "CZ", "CZ2", "CZ3", "CH2"];

    let mut best_match: Option<(String, String, f64)> = None;

    for carbon_type in carbon_types {
        // Collect all BMRB entries for this carbon type
        let matching_c_stats: Vec<_> = all_carbons
            .iter()
            .filter(|s| s.atom_name == carbon_type)
            .collect();

        if matching_c_stats.is_empty() {
            continue;
        }

        // Calculate average mean and std_dev for this carbon type
        let c_mean: f64 = matching_c_stats.iter().map(|s| s.mean).sum::<f64>() / matching_c_stats.len() as f64;
        let c_std: f64 = matching_c_stats.iter().map(|s| s.std_dev).sum::<f64>() / matching_c_stats.len() as f64;
        let c_std = c_std.max(1.0); // Min std dev for carbons

        // Calculate carbon z-score
        let c_zscore = ((c_shift - c_mean) / c_std).abs();

        // Get the corresponding proton type
        let proton_type = carbon_to_proton_name(carbon_type);

        // Get proton statistics
        let all_protons = bmrb.get_nucleus_atoms("H");
        let matching_h_stats: Vec<_> = all_protons
            .iter()
            .filter(|s| s.atom_name == proton_type || s.atom_name == format!("{}2", proton_type) || s.atom_name == format!("{}3", proton_type))
            .collect();

        if matching_h_stats.is_empty() {
            continue;
        }

        // Calculate average mean and std_dev for the proton type
        let h_mean: f64 = matching_h_stats.iter().map(|s| s.mean).sum::<f64>() / matching_h_stats.len() as f64;
        let h_std: f64 = matching_h_stats.iter().map(|s| s.std_dev).sum::<f64>() / matching_h_stats.len() as f64;
        let h_std = h_std.max(0.1); // Min std dev for protons

        // Calculate proton z-score
        let h_zscore = ((h_shift - h_mean) / h_std).abs();

        // Combined score (average of carbon and proton z-scores)
        let combined_score = (c_zscore + h_zscore) / 2.0;

        // Only consider if both are within 3 sigma
        if c_zscore < 3.0 && h_zscore < 3.0 {
            if best_match.is_none() || combined_score < best_match.as_ref().unwrap().2 {
                let normalized_c = normalize_carbon_name(carbon_type);
                let normalized_h = normalize_proton_name(&proton_type);
                best_match = Some((normalized_c, normalized_h, combined_score));
            }
        }
    }

    best_match.map(|(c, h, _)| (c, h))
}

/// Map carbon atom name to corresponding proton atom name.
fn carbon_to_proton_name(carbon: &str) -> String {
    match carbon {
        "CA" => "HA".to_string(),
        "CB" => "HB".to_string(),
        "CG" | "CG1" | "CG2" => "HG".to_string(),
        "CD" | "CD1" | "CD2" => "HD".to_string(),
        "CE" | "CE1" | "CE2" => "HE".to_string(),
        "CE3" => "HE3".to_string(),   // TRP aromatic
        "CZ" => "HZ".to_string(),
        "CZ2" => "HZ2".to_string(),   // TRP aromatic
        "CZ3" => "HZ3".to_string(),   // TRP aromatic
        "CH2" => "HH2".to_string(),   // TRP aromatic
        _ => "HX".to_string(),
    }
}

/// Normalize carbon atom names to canonical form.
/// Note: Aromatic-specific carbons (CE3, CZ2, CZ3, CH2) are kept distinct.
fn normalize_carbon_name(carbon_name: &str) -> String {
    match carbon_name {
        "CG1" | "CG2" => "CG".to_string(),
        // Don't normalize aromatic carbons from PHE/TYR/TRP - they're distinct
        // CD1/CD2 at ~131 ppm (aromatic) vs CD1/CD2 at ~13-25 ppm (aliphatic) are VERY different
        // Keep them as-is for now since the shift will disambiguate
        "CD1" | "CD2" => "CD".to_string(),
        "CE1" | "CE2" => "CE".to_string(),
        // TRP-specific aromatic carbons - keep distinct
        "CE3" | "CZ2" | "CZ3" | "CH2" => carbon_name.to_string(),
        _ => carbon_name.to_string(),
    }
}

/// Assign carbon shifts from 13C-HSQC by matching proton positions (fallback).
#[allow(dead_code)]
fn assign_carbons_from_hsqc(
    ss: &mut SpinSystem,
    hsqc_13c_peaks: &[UnlabeledPeak],
    tocsy_correlations: &[(f64, f64)],
    params: &SpinSystemBuildParams,
) {
    // Collect all proton positions in this spin system
    let mut proton_positions: Vec<(String, f64)> = ss
        .proton_shifts
        .iter()
        .map(|(name, &shift)| (name.clone(), shift))
        .collect();

    // Also include TOCSY correlations that weren't classified
    for (h_shift, _) in tocsy_correlations {
        let already_added = proton_positions.iter().any(|(_, s)| (*s - h_shift).abs() < 0.02);
        if !already_added {
            proton_positions.push(("unclassified".to_string(), *h_shift));
        }
    }

    // Match 13C-HSQC peaks to protons
    for peak in hsqc_13c_peaks {
        if peak.experiment_type != PeakExperimentType::Hsqc13C {
            continue;
        }

        let c_shift = peak.position_ppm[0];
        let h_shift = peak.position_ppm[1];

        // Find matching proton
        for (proton_name, proton_shift) in &proton_positions {
            if (h_shift - proton_shift).abs() < params.h_tolerance {
                // Determine carbon name from the carbon shift
                let carbon_name = infer_carbon_name(c_shift, proton_name);

                // Only add if we don't already have this carbon
                if !ss.carbon_shifts.contains_key(&carbon_name) {
                    ss.add_carbon_shift(&carbon_name, c_shift);

                    // If proton was unclassified, try to name it now
                    if proton_name == "unclassified" {
                        let h_name = infer_proton_name(&carbon_name);
                        if !ss.proton_shifts.contains_key(&h_name) {
                            ss.add_proton_shift(&h_name, *proton_shift);
                        }
                    }
                }
                break;
            }
        }
    }
}

/// Infer carbon atom name from chemical shift and attached proton name.
fn infer_carbon_name(c_shift: f64, proton_name: &str) -> String {
    // If we know the proton, use that to guide carbon naming
    match proton_name {
        "HA" | "HA2" | "HA3" => "CA".to_string(),
        "HB" | "HB2" | "HB3" => "CB".to_string(),
        "HG" | "HG2" | "HG3" | "HG12" | "HG13" => {
            // Check if it's more likely CG1 or CG2 based on shift
            if c_shift < 20.0 {
                "CG2".to_string() // Methyl gamma (VAL, ILE)
            } else {
                "CG".to_string()
            }
        }
        "HD" | "HD2" | "HD3" | "HD1" => "CD".to_string(),
        "HE" | "HE2" | "HE3" | "HE1" => "CE".to_string(),
        _ => {
            // Fall back to shift-based classification
            if c_shift < 25.0 {
                "CB".to_string() // Methyl CB (ALA)
            } else if c_shift < 35.0 {
                "CB".to_string()
            } else if c_shift < 45.0 {
                "CG".to_string()
            } else if c_shift < 55.0 {
                "CB".to_string()
            } else if c_shift < 70.0 {
                "CA".to_string()
            } else {
                "CX".to_string() // Unknown
            }
        }
    }
}

/// Infer proton name from carbon name.
fn infer_proton_name(carbon_name: &str) -> String {
    match carbon_name {
        "CA" => "HA".to_string(),
        "CB" => "HB".to_string(),
        "CG" | "CG1" | "CG2" => "HG".to_string(),
        "CD" | "CD1" | "CD2" => "HD".to_string(),
        "CE" | "CE1" | "CE2" => "HE".to_string(),
        _ => "HX".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_peaks() -> (Vec<UnlabeledPeak>, Vec<UnlabeledPeak>, Vec<UnlabeledPeak>) {
        // Simulate an ALA residue
        // 15N-HSQC: N=123.5, H=8.21
        let hsqc_15n = vec![UnlabeledPeak::hsqc_15n(123.5, 8.21, 1000.0)];

        // 13C-HSQC: CA=52.5/HA=4.35, CB=19.1/HB=1.42
        let hsqc_13c = vec![
            UnlabeledPeak::hsqc_13c(52.5, 4.35, 800.0), // CA-HA
            UnlabeledPeak::hsqc_13c(19.1, 1.42, 600.0), // CB-HB
        ];

        // TOCSY from H=8.21: correlations to HA=4.35, HB=1.42
        let tocsy = vec![
            UnlabeledPeak::tocsy(8.21, 4.35, 500.0), // H-HA
            UnlabeledPeak::tocsy(8.21, 1.42, 400.0), // H-HB
            UnlabeledPeak::tocsy(4.35, 8.21, 500.0), // Symmetric HA-H
            UnlabeledPeak::tocsy(1.42, 8.21, 400.0), // Symmetric HB-H
            UnlabeledPeak::tocsy(4.35, 1.42, 300.0), // HA-HB
        ];

        (hsqc_15n, hsqc_13c, tocsy)
    }

    #[test]
    fn test_build_spin_system_basic() {
        let (hsqc_15n, hsqc_13c, tocsy) = make_test_peaks();
        let params = SpinSystemBuildParams::default();

        let spin_systems = build_spin_systems(&hsqc_15n, &hsqc_13c, &tocsy, &params);

        assert_eq!(spin_systems.len(), 1);

        let ss = &spin_systems[0];
        assert!((ss.h_shift - 8.21).abs() < 0.01);
        assert!((ss.n_shift - 123.5).abs() < 0.1);

        // Should have found TOCSY correlations
        assert!(
            ss.proton_shifts.len() >= 1,
            "Expected at least 1 proton shift, got {}",
            ss.proton_shifts.len()
        );
    }

    #[test]
    fn test_tocsy_correlation_finding() {
        let tocsy = vec![
            UnlabeledPeak::tocsy(8.21, 4.35, 500.0),
            UnlabeledPeak::tocsy(8.21, 1.42, 400.0),
            UnlabeledPeak::tocsy(8.21, 8.21, 1000.0), // Diagonal - should be ignored
        ];

        let correlations = find_tocsy_correlations(8.21, &tocsy, 0.03, 0.05);

        // Should find 4.35 and 1.42, not diagonal
        assert_eq!(correlations.len(), 2);
        assert!(correlations.iter().any(|(h, _)| (*h - 4.35).abs() < 0.01));
        assert!(correlations.iter().any(|(h, _)| (*h - 1.42).abs() < 0.01));
    }

    #[test]
    fn test_proton_classification() {
        // HA region
        let ha = classify_proton_shift(4.35, 8.2);
        assert_eq!(ha, Some("HA".to_string()));

        // HB region (aliphatic methyl)
        let hb = classify_proton_shift(1.42, 8.2);
        assert!(hb.is_some()); // Should classify as some sidechain proton

        // Backbone H itself - should be None
        let backbone = classify_proton_shift(8.21, 8.2);
        assert!(backbone.is_none());
    }

    #[test]
    fn test_carbon_classification() {
        // CA-HA region
        let ca = classify_carbon_hsqc_peak(52.5, 4.35);
        assert!(ca.is_some());
        let (c_name, h_name) = ca.unwrap();
        assert_eq!(c_name, "CA");
        assert_eq!(h_name, "HA");

        // CB-HB ALA (methyl)
        let cb = classify_carbon_hsqc_peak(19.1, 1.42);
        assert!(cb.is_some());
        let (c_name, _) = cb.unwrap();
        assert_eq!(c_name, "CB");
    }

    #[test]
    fn test_multiple_spin_systems() {
        // Two residues
        let hsqc_15n = vec![
            UnlabeledPeak::hsqc_15n(123.5, 8.21, 1000.0),
            UnlabeledPeak::hsqc_15n(119.8, 8.05, 900.0),
        ];

        let hsqc_13c = vec![
            UnlabeledPeak::hsqc_13c(52.5, 4.35, 800.0),
            UnlabeledPeak::hsqc_13c(55.2, 4.28, 750.0),
        ];

        let tocsy = vec![
            UnlabeledPeak::tocsy(8.21, 4.35, 500.0),
            UnlabeledPeak::tocsy(8.05, 4.28, 480.0),
        ];

        let params = SpinSystemBuildParams::default();
        let spin_systems = build_spin_systems(&hsqc_15n, &hsqc_13c, &tocsy, &params);

        assert_eq!(spin_systems.len(), 2);
    }
}
