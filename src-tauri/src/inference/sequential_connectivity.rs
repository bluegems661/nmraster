//! Sequential connectivity detection from NOESY data.
//!
//! This module identifies sequential relationships between spin systems
//! by analyzing NOESY cross-peaks that indicate through-space proximity.
//!
//! Key NOE types for sequential assignment:
//! - dαN(i,i+1): HA(i) to H(i+1) - most reliable, ~2.2-3.5 Å
//! - dNN(i,i+1): H(i) to H(i+1) - present in helices, ~2.8-4.5 Å
//! - dβN(i,i+1): HB(i) to H(i+1) - additional evidence

use std::collections::HashMap;
use uuid::Uuid;

use crate::data::{NOEType, PeakExperimentType, SequentialLink, SpinSystem, UnlabeledPeak};

/// Parameters for sequential connectivity detection.
#[derive(Debug, Clone)]
pub struct ConnectivityParams {
    /// Tolerance for matching proton positions (ppm)
    pub h_tolerance: f64,
    /// Minimum confidence to report a link
    pub min_confidence: f64,
    /// Weight for dαN evidence (most reliable)
    pub dan_weight: f64,
    /// Weight for dNN evidence (helices)
    pub dnn_weight: f64,
    /// Weight for dβN evidence (additional)
    pub dbn_weight: f64,
}

impl Default for ConnectivityParams {
    fn default() -> Self {
        Self {
            h_tolerance: 0.03,
            min_confidence: 0.3,
            dan_weight: 1.0,   // dαN is most reliable
            dnn_weight: 0.7,   // dNN is helpful but less specific
            dbn_weight: 0.5,   // dβN is additional evidence
        }
    }
}

/// Find sequential links between spin systems from NOESY data.
///
/// For each pair of spin systems, looks for NOE evidence of sequential
/// connectivity (i → i+1 relationship).
pub fn find_sequential_links(
    spin_systems: &[SpinSystem],
    noesy_peaks: &[UnlabeledPeak],
    params: &ConnectivityParams,
) -> Vec<SequentialLink> {
    let mut links = Vec::new();

    // Build index of spin systems by H shift for fast lookup
    let h_index: HashMap<usize, f64> = spin_systems
        .iter()
        .enumerate()
        .map(|(i, ss)| (i, ss.h_shift))
        .collect();

    // For each potential i → i+1 pair
    for (i, ss_i) in spin_systems.iter().enumerate() {
        for (j, ss_j) in spin_systems.iter().enumerate() {
            if i == j {
                continue;
            }

            // Look for evidence that ss_i precedes ss_j (ss_i is i, ss_j is i+1)
            let (confidence, noe_type, supporting_peaks) =
                evaluate_sequential_evidence(ss_i, ss_j, noesy_peaks, params);

            if confidence >= params.min_confidence {
                let mut link = SequentialLink::new(ss_i.id, ss_j.id, noe_type, confidence);
                for peak_id in supporting_peaks {
                    link.add_supporting_peak(peak_id);
                }
                links.push(link);
            }
        }
    }

    // Sort by confidence (highest first)
    links.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    tracing::info!(
        "Found {} sequential links from {} spin systems",
        links.len(),
        spin_systems.len()
    );

    links
}

/// Evaluate evidence for ss_from preceding ss_to in sequence.
///
/// Returns (confidence, dominant NOE type, supporting peak IDs).
fn evaluate_sequential_evidence(
    ss_from: &SpinSystem,  // Position i (preceding)
    ss_to: &SpinSystem,    // Position i+1 (following)
    noesy_peaks: &[UnlabeledPeak],
    params: &ConnectivityParams,
) -> (f64, NOEType, Vec<Uuid>) {
    let mut total_score = 0.0;
    let mut supporting_peaks = Vec::new();
    let mut dominant_type = NOEType::Other;
    let mut best_score = 0.0;

    // Check for dαN: H(i+1) ↔ HA(i)
    // ss_to is i+1, ss_from is i
    if let Some(&ha_i) = ss_from.proton_shifts.get("HA") {
        let h_i1 = ss_to.h_shift;

        let dan_evidence = find_noe_between(ha_i, h_i1, noesy_peaks, params.h_tolerance);
        if !dan_evidence.is_empty() {
            let score = params.dan_weight * (1.0 + 0.2 * (dan_evidence.len() - 1) as f64);
            total_score += score;
            supporting_peaks.extend(dan_evidence);

            if score > best_score {
                best_score = score;
                dominant_type = NOEType::DAlphaN;
            }
        }
    }

    // Check for dNN: H(i+1) ↔ H(i)
    let h_i = ss_from.h_shift;
    let h_i1 = ss_to.h_shift;

    let dnn_evidence = find_noe_between(h_i, h_i1, noesy_peaks, params.h_tolerance);
    if !dnn_evidence.is_empty() {
        let score = params.dnn_weight * (1.0 + 0.2 * (dnn_evidence.len() - 1) as f64);
        total_score += score;
        supporting_peaks.extend(dnn_evidence);

        if score > best_score {
            best_score = score;
            dominant_type = NOEType::DNN;
        }
    }

    // Check for dβN: H(i+1) ↔ HB(i)
    if let Some(&hb_i) = ss_from.proton_shifts.get("HB") {
        let dbn_evidence = find_noe_between(hb_i, h_i1, noesy_peaks, params.h_tolerance);
        if !dbn_evidence.is_empty() {
            let score = params.dbn_weight * (1.0 + 0.2 * (dbn_evidence.len() - 1) as f64);
            total_score += score;
            supporting_peaks.extend(dbn_evidence);

            if score > best_score {
                best_score = score;
                dominant_type = NOEType::DBetaN;
            }
        }
    }

    // Also check HB2/HB3 for methylenes
    for hb_name in &["HB2", "HB3"] {
        if let Some(&hb_i) = ss_from.proton_shifts.get(*hb_name) {
            let dbn_evidence = find_noe_between(hb_i, h_i1, noesy_peaks, params.h_tolerance);
            if !dbn_evidence.is_empty() {
                let score = params.dbn_weight * 0.7 * (1.0 + 0.1 * (dbn_evidence.len() - 1) as f64);
                total_score += score;
                supporting_peaks.extend(dbn_evidence);
            }
        }
    }

    // Normalize confidence to 0-1 range
    // Max possible score is roughly dan + dnn + dbn = ~2.2
    let confidence = (total_score / 2.5).min(1.0);

    (confidence, dominant_type, supporting_peaks)
}

/// Find NOESY peaks connecting two proton positions.
///
/// Returns the IDs of matching NOESY peaks.
fn find_noe_between(
    h1: f64,
    h2: f64,
    noesy_peaks: &[UnlabeledPeak],
    tolerance: f64,
) -> Vec<Uuid> {
    let mut matches = Vec::new();

    for peak in noesy_peaks {
        if peak.experiment_type != PeakExperimentType::Noesy {
            continue;
        }

        let p1 = peak.position_ppm[0];
        let p2 = peak.position_ppm[1];

        // NOESY is symmetric - check both orientations
        let match_forward = (p1 - h1).abs() < tolerance && (p2 - h2).abs() < tolerance;
        let match_reverse = (p1 - h2).abs() < tolerance && (p2 - h1).abs() < tolerance;

        if match_forward || match_reverse {
            matches.push(peak.id);
        }
    }

    matches
}

/// Build a connectivity graph from sequential links.
///
/// Returns adjacency lists for forward and backward connections.
pub fn build_connectivity_graph(
    spin_systems: &[SpinSystem],
    links: &[SequentialLink],
) -> (HashMap<Uuid, Vec<Uuid>>, HashMap<Uuid, Vec<Uuid>>) {
    let mut forward: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
    let mut backward: HashMap<Uuid, Vec<Uuid>> = HashMap::new();

    // Initialize with all spin systems
    for ss in spin_systems {
        forward.insert(ss.id, Vec::new());
        backward.insert(ss.id, Vec::new());
    }

    // Add edges from links
    for link in links {
        forward
            .entry(link.from_spin_system)
            .or_default()
            .push(link.to_spin_system);
        backward
            .entry(link.to_spin_system)
            .or_default()
            .push(link.from_spin_system);
    }

    (forward, backward)
}

/// Find chains of sequentially connected spin systems.
///
/// Returns vectors of spin system IDs representing contiguous stretches.
pub fn find_sequential_chains(
    spin_systems: &[SpinSystem],
    links: &[SequentialLink],
    min_chain_length: usize,
) -> Vec<Vec<Uuid>> {
    let (forward, backward) = build_connectivity_graph(spin_systems, links);

    // Find chain starts (spin systems with no backward links but forward links)
    let chain_starts: Vec<Uuid> = spin_systems
        .iter()
        .filter(|ss| {
            backward.get(&ss.id).map(|v| v.is_empty()).unwrap_or(true)
                && forward.get(&ss.id).map(|v| !v.is_empty()).unwrap_or(false)
        })
        .map(|ss| ss.id)
        .collect();

    let mut chains = Vec::new();

    for start in chain_starts {
        let chain = trace_chain(&start, &forward);
        if chain.len() >= min_chain_length {
            chains.push(chain);
        }
    }

    // Sort by length (longest first)
    chains.sort_by(|a, b| b.len().cmp(&a.len()));

    tracing::info!(
        "Found {} sequential chains (min length {}), longest: {}",
        chains.len(),
        min_chain_length,
        chains.first().map(|c| c.len()).unwrap_or(0)
    );

    chains
}

/// Trace a chain from a starting spin system.
fn trace_chain(start: &Uuid, forward: &HashMap<Uuid, Vec<Uuid>>) -> Vec<Uuid> {
    let mut chain = vec![*start];
    let mut current = *start;
    let mut visited = std::collections::HashSet::new();
    visited.insert(current);

    loop {
        let next_options = forward.get(&current);
        match next_options {
            Some(options) if !options.is_empty() => {
                // Take the first unvisited option
                // In practice, might want to choose based on confidence
                if let Some(&next) = options.iter().find(|id| !visited.contains(id)) {
                    chain.push(next);
                    visited.insert(next);
                    current = next;
                } else {
                    break;
                }
            }
            _ => break,
        }
    }

    chain
}

/// Get connectivity statistics for diagnostics.
pub fn connectivity_statistics(
    spin_systems: &[SpinSystem],
    links: &[SequentialLink],
) -> ConnectivityStats {
    let (forward, backward) = build_connectivity_graph(spin_systems, links);

    let isolated = spin_systems
        .iter()
        .filter(|ss| {
            forward.get(&ss.id).map(|v| v.is_empty()).unwrap_or(true)
                && backward.get(&ss.id).map(|v| v.is_empty()).unwrap_or(true)
        })
        .count();

    let chain_ends = spin_systems
        .iter()
        .filter(|ss| {
            forward.get(&ss.id).map(|v| v.is_empty()).unwrap_or(true)
                && backward.get(&ss.id).map(|v| !v.is_empty()).unwrap_or(false)
        })
        .count();

    let chain_starts = spin_systems
        .iter()
        .filter(|ss| {
            backward.get(&ss.id).map(|v| v.is_empty()).unwrap_or(true)
                && forward.get(&ss.id).map(|v| !v.is_empty()).unwrap_or(false)
        })
        .count();

    let avg_confidence = if links.is_empty() {
        0.0
    } else {
        links.iter().map(|l| l.confidence).sum::<f64>() / links.len() as f64
    };

    ConnectivityStats {
        total_spin_systems: spin_systems.len(),
        total_links: links.len(),
        isolated_spin_systems: isolated,
        chain_starts,
        chain_ends,
        average_confidence: avg_confidence,
    }
}

/// Statistics about sequential connectivity.
#[derive(Debug, Clone)]
pub struct ConnectivityStats {
    pub total_spin_systems: usize,
    pub total_links: usize,
    pub isolated_spin_systems: usize,
    pub chain_starts: usize,
    pub chain_ends: usize,
    pub average_confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_spin_systems() -> Vec<SpinSystem> {
        // Create 3 spin systems representing a tripeptide
        let anchor1 = UnlabeledPeak::hsqc_15n(123.5, 8.21, 1000.0);
        let anchor2 = UnlabeledPeak::hsqc_15n(119.8, 8.05, 900.0);
        let anchor3 = UnlabeledPeak::hsqc_15n(121.2, 7.95, 850.0);

        let mut ss1 = SpinSystem::new(&anchor1);
        ss1.add_proton_shift("HA", 4.35);
        ss1.add_proton_shift("HB", 1.42);

        let mut ss2 = SpinSystem::new(&anchor2);
        ss2.add_proton_shift("HA", 4.28);
        ss2.add_proton_shift("HB", 2.95);

        let mut ss3 = SpinSystem::new(&anchor3);
        ss3.add_proton_shift("HA", 4.15);
        ss3.add_proton_shift("HB", 1.85);

        vec![ss1, ss2, ss3]
    }

    fn make_test_noesy_peaks(spin_systems: &[SpinSystem]) -> Vec<UnlabeledPeak> {
        let mut peaks = Vec::new();

        // dαN: HA(1) to H(2)
        let ha1 = spin_systems[0].proton_shifts.get("HA").unwrap();
        let h2 = spin_systems[1].h_shift;
        peaks.push(UnlabeledPeak::noesy(*ha1, h2, 500.0));

        // dNN: H(1) to H(2)
        let h1 = spin_systems[0].h_shift;
        peaks.push(UnlabeledPeak::noesy(h1, h2, 300.0));

        // dαN: HA(2) to H(3)
        let ha2 = spin_systems[1].proton_shifts.get("HA").unwrap();
        let h3 = spin_systems[2].h_shift;
        peaks.push(UnlabeledPeak::noesy(*ha2, h3, 480.0));

        peaks
    }

    #[test]
    fn test_find_sequential_links() {
        let spin_systems = make_test_spin_systems();
        let noesy_peaks = make_test_noesy_peaks(&spin_systems);
        let params = ConnectivityParams::default();

        let links = find_sequential_links(&spin_systems, &noesy_peaks, &params);

        // Should find links: ss1→ss2 and ss2→ss3
        assert!(links.len() >= 2, "Expected at least 2 links, got {}", links.len());

        // Check that we found ss1 → ss2
        let ss1_to_ss2 = links.iter().find(|l| {
            l.from_spin_system == spin_systems[0].id && l.to_spin_system == spin_systems[1].id
        });
        assert!(ss1_to_ss2.is_some(), "Should find link from ss1 to ss2");
    }

    #[test]
    fn test_find_noe_between() {
        let peaks = vec![
            UnlabeledPeak::noesy(8.21, 4.35, 500.0),
            UnlabeledPeak::noesy(8.05, 4.28, 400.0),
        ];

        let matches = find_noe_between(8.21, 4.35, &peaks, 0.03);
        assert_eq!(matches.len(), 1);

        // Check symmetric matching
        let matches_rev = find_noe_between(4.35, 8.21, &peaks, 0.03);
        assert_eq!(matches_rev.len(), 1);

        // No match for different positions
        let no_match = find_noe_between(7.5, 3.0, &peaks, 0.03);
        assert!(no_match.is_empty());
    }

    #[test]
    fn test_find_sequential_chains() {
        let spin_systems = make_test_spin_systems();
        let noesy_peaks = make_test_noesy_peaks(&spin_systems);
        let params = ConnectivityParams::default();

        let links = find_sequential_links(&spin_systems, &noesy_peaks, &params);
        let chains = find_sequential_chains(&spin_systems, &links, 2);

        // Should find at least one chain of length 2+
        assert!(!chains.is_empty(), "Should find at least one chain");
        assert!(
            chains[0].len() >= 2,
            "Longest chain should be at least 2, got {}",
            chains[0].len()
        );
    }

    #[test]
    fn test_connectivity_statistics() {
        let spin_systems = make_test_spin_systems();
        let noesy_peaks = make_test_noesy_peaks(&spin_systems);
        let params = ConnectivityParams::default();

        let links = find_sequential_links(&spin_systems, &noesy_peaks, &params);
        let stats = connectivity_statistics(&spin_systems, &links);

        assert_eq!(stats.total_spin_systems, 3);
        assert!(stats.total_links >= 2);
        assert!(stats.average_confidence > 0.0);
    }
}
