//! Integration tests for unified NMR assignment.
//!
//! Tests the unified approach: all evidence (TOCSY, carbon typing, NOESY)
//! processed simultaneously in one factor graph using KDE-based scoring.

#[cfg(test)]
mod tests {
    use crate::data::{PeakExperimentType, UnlabeledPeak};
    use crate::inference::unified_assignment::{run_unified_assignment, UnifiedAssignmentParams, PeakType};
    use crate::testdata::{BMRBDatabase, KDEDatabase};

    /// Get C-H pairs for 13C-HSQC peaks using BMRB/NEF atom naming.
    fn get_ch_pairs(aa: &str) -> Vec<(&'static str, &'static str)> {
        match aa {
            "GLY" => vec![("CA", "HA2")],
            "ALA" => vec![("CA", "HA"), ("CB", "HB1")],
            "VAL" => vec![("CA", "HA"), ("CB", "HB"), ("CG1", "HG11"), ("CG2", "HG21")],
            "LEU" => vec![("CA", "HA"), ("CB", "HB2"), ("CG", "HG"), ("CD1", "HD11"), ("CD2", "HD21")],
            "ILE" => vec![("CA", "HA"), ("CB", "HB"), ("CG1", "HG12"), ("CG2", "HG21"), ("CD1", "HD11")],
            "PRO" => vec![("CA", "HA"), ("CB", "HB2"), ("CG", "HG2"), ("CD", "HD2")],
            "MET" => vec![("CA", "HA"), ("CB", "HB2"), ("CG", "HG2"), ("CE", "HE1")],
            "PHE" => vec![("CA", "HA"), ("CB", "HB2"), ("CD1", "HD1"), ("CE1", "HE1"), ("CZ", "HZ")],
            "TYR" => vec![("CA", "HA"), ("CB", "HB2"), ("CD1", "HD1"), ("CE1", "HE1")],
            "TRP" => vec![("CA", "HA"), ("CB", "HB2"), ("CD1", "HD1"), ("CE3", "HE3"), ("CZ2", "HZ2"), ("CZ3", "HZ3"), ("CH2", "HH2")],
            "HIS" => vec![("CA", "HA"), ("CB", "HB2"), ("CD2", "HD2"), ("CE1", "HE1")],
            "SER" => vec![("CA", "HA"), ("CB", "HB2")],
            "THR" => vec![("CA", "HA"), ("CB", "HB"), ("CG2", "HG21")],
            "CYS" => vec![("CA", "HA"), ("CB", "HB2")],
            "ASN" => vec![("CA", "HA"), ("CB", "HB2")],
            "GLN" => vec![("CA", "HA"), ("CB", "HB2"), ("CG", "HG2")],
            "ASP" => vec![("CA", "HA"), ("CB", "HB2")],
            "GLU" => vec![("CA", "HA"), ("CB", "HB2"), ("CG", "HG2")],
            "LYS" => vec![("CA", "HA"), ("CB", "HB2"), ("CG", "HG2"), ("CD", "HD2"), ("CE", "HE2")],
            "ARG" => vec![("CA", "HA"), ("CB", "HB2"), ("CG", "HG2"), ("CD", "HD2")],
            _ => vec![],
        }
    }

    /// Get all protons in a spin system for TOCSY correlations.
    fn get_spin_system_protons(aa: &str) -> Vec<&'static str> {
        match aa {
            "GLY" => vec!["HA2"],
            "ALA" => vec!["HA", "HB1"],
            "VAL" => vec!["HA", "HB", "HG11", "HG21"],
            "LEU" => vec!["HA", "HB2", "HG", "HD11", "HD21"],
            "ILE" => vec!["HA", "HB", "HG12", "HG21", "HD11"],
            "PRO" => vec!["HA", "HB2", "HG2", "HD2"],
            "MET" => vec!["HA", "HB2", "HG2", "HE1"],
            "PHE" => vec!["HA", "HB2", "HD1", "HE1", "HZ"],
            "TYR" => vec!["HA", "HB2", "HD1", "HE1"],
            "TRP" => vec!["HA", "HB2", "HD1", "HE3", "HZ2", "HZ3", "HH2"],
            "HIS" => vec!["HA", "HB2", "HD2", "HE1"],
            "SER" => vec!["HA", "HB2"],
            "THR" => vec!["HA", "HB", "HG21"],
            "CYS" => vec!["HA", "HB2"],
            "ASN" => vec!["HA", "HB2"],
            "GLN" => vec!["HA", "HB2", "HG2"],
            "ASP" => vec!["HA", "HB2"],
            "GLU" => vec!["HA", "HB2", "HG2"],
            "LYS" => vec!["HA", "HB2", "HG2", "HD2", "HE2"],
            "ARG" => vec!["HA", "HB2", "HG2", "HD2"],
            _ => vec![],
        }
    }

    fn one_letter_to_three(c: &char) -> String {
        match c {
            'A' => "ALA", 'C' => "CYS", 'D' => "ASP", 'E' => "GLU", 'F' => "PHE",
            'G' => "GLY", 'H' => "HIS", 'I' => "ILE", 'K' => "LYS", 'L' => "LEU",
            'M' => "MET", 'N' => "ASN", 'P' => "PRO", 'Q' => "GLN", 'R' => "ARG",
            'S' => "SER", 'T' => "THR", 'V' => "VAL", 'W' => "TRP", 'Y' => "TYR",
            _ => "UNK",
        }.to_string()
    }

    /// Generate test peaks using exact KDE mode values (most likely shifts).
    fn generate_idealized_peaks(sequence: &str) -> (
        Vec<UnlabeledPeak>,  // 15N-HSQC
        Vec<UnlabeledPeak>,  // 13C-HSQC
        Vec<UnlabeledPeak>,  // TOCSY
        Vec<UnlabeledPeak>,  // NOESY
        Vec<(i32, String, f64, f64)>,  // Ground truth: (seq_code, name, H, N)
    ) {
        let kde = KDEDatabase::load_embedded();
        let residues: Vec<char> = sequence.chars().collect();
        let n = residues.len();

        let mut hsqc_15n = Vec::new();
        let mut hsqc_13c = Vec::new();
        let mut tocsy = Vec::new();
        let mut noesy = Vec::new();
        let mut ground_truth = Vec::new();

        struct ResidueData {
            h_backbone: f64,
        }
        let mut residue_data: Vec<Option<ResidueData>> = Vec::new();

        for (i, &aa) in residues.iter().enumerate() {
            let seq_code = (i + 1) as i32;
            let aa_name = one_letter_to_three(&aa);

            // Skip proline (no backbone NH)
            if aa == 'P' {
                residue_data.push(None);
                continue;
            }

            // Backbone H and N: spread out for unique 15N-HSQC peaks
            let h_backbone = 7.5 + (i as f64) * 2.0 / (n as f64 - 1.0).max(1.0);
            let n_backbone = 115.0 + (i as f64) * 15.0 / (n as f64 - 1.0).max(1.0);

            // 15N-HSQC peak
            hsqc_15n.push(UnlabeledPeak::hsqc_15n(n_backbone, h_backbone, 1.0));

            // 13C-HSQC peaks: use KDE modes for each C-H pair
            let ch_pairs = get_ch_pairs(&aa_name);
            for (carbon, proton) in ch_pairs {
                let c_shift = kde.get(&aa_name, carbon).map(|g| g.mode()).unwrap_or(55.0);
                let h_shift = kde.get(&aa_name, proton).map(|g| g.mode()).unwrap_or(4.0);
                hsqc_13c.push(UnlabeledPeak::hsqc_13c(c_shift, h_shift, 1.0));
            }

            // TOCSY: correlations between backbone H and all protons in spin system
            let spin_protons = get_spin_system_protons(&aa_name);
            for proton in spin_protons {
                if let Some(grid) = kde.get(&aa_name, proton) {
                    let h_shift = grid.mode();
                    tocsy.push(UnlabeledPeak::tocsy(h_backbone, h_shift, 1.0));
                    tocsy.push(UnlabeledPeak::tocsy(h_shift, h_backbone, 1.0));

                    if proton != "HA" && proton != "HA2" {
                        if let Some(ha_grid) = kde.get(&aa_name, "HA").or_else(|| kde.get(&aa_name, "HA2")) {
                            let ha_shift = ha_grid.mode();
                            tocsy.push(UnlabeledPeak::tocsy(ha_shift, h_shift, 0.8));
                            tocsy.push(UnlabeledPeak::tocsy(h_shift, ha_shift, 0.8));
                        }
                    }
                }
            }

            residue_data.push(Some(ResidueData { h_backbone }));
            ground_truth.push((seq_code, aa_name, h_backbone, n_backbone));
        }

        // NOESY: sequential dαN (H(i) to HA(i-1))
        for i in 1..residues.len() {
            if residues[i] == 'P' || residues[i-1] == 'P' { continue; }
            let aa_prev = one_letter_to_three(&residues[i-1]);

            if let (Some(curr), Some(_prev)) = (&residue_data[i], &residue_data[i-1]) {
                if let Some(ha_grid) = kde.get(&aa_prev, "HA").or_else(|| kde.get(&aa_prev, "HA2")) {
                    let ha_prev = ha_grid.mode();
                    noesy.push(UnlabeledPeak::noesy(curr.h_backbone, ha_prev, 0.5));
                    noesy.push(UnlabeledPeak::noesy(ha_prev, curr.h_backbone, 0.5));
                }
            }
        }

        (hsqc_15n, hsqc_13c, tocsy, noesy, ground_truth)
    }

    /// Generate peaks with REALISTIC BMRB proton shifts (overlapping HA/HB).
    fn generate_realistic_bmrb_peaks(sequence: &str) -> (
        Vec<UnlabeledPeak>,
        Vec<UnlabeledPeak>,
        Vec<UnlabeledPeak>,
        Vec<UnlabeledPeak>,
        Vec<(i32, String, f64, f64)>,
    ) {
        let bmrb = BMRBDatabase::load_embedded();
        let residues: Vec<char> = sequence.chars().collect();
        let n = residues.len();

        let mut hsqc_15n = Vec::new();
        let mut hsqc_13c = Vec::new();
        let mut tocsy = Vec::new();
        let mut noesy = Vec::new();
        let mut ground_truth = Vec::new();
        let mut residue_shifts: Vec<(f64, f64, f64)> = Vec::new();

        for (i, &aa) in residues.iter().enumerate() {
            let seq_code = (i + 1) as i32;
            let aa_name = one_letter_to_three(&aa);

            if aa == 'P' {
                residue_shifts.push((0.0, 0.0, 0.0));
                continue;
            }

            let h_backbone = 7.5 + (i as f64) * 2.0 / (n as f64 - 1.0).max(1.0);
            let n_backbone = 115.0 + (i as f64) * 15.0 / (n as f64 - 1.0).max(1.0);
            let ha_shift = bmrb.get(&aa_name, "HA").map(|s| s.mean).unwrap_or(4.3);
            let ca_shift = bmrb.get(&aa_name, "CA").map(|s| s.mean).unwrap_or(55.0);

            let sidechain_atoms = [
                ("CB", "HB"), ("CG", "HG"), ("CG1", "HG1"), ("CG2", "HG2"),
                ("CD", "HD"), ("CD1", "HD1"), ("CD2", "HD2"),
                ("CE", "HE"), ("CE1", "HE1"), ("CE2", "HE2"), ("CZ", "HZ"),
            ];

            let mut sidechain_peaks: Vec<(f64, f64)> = Vec::new();
            for (c_atom, h_atom) in &sidechain_atoms {
                if let (Some(c_stats), Some(h_stats)) = (bmrb.get(&aa_name, c_atom), bmrb.get(&aa_name, h_atom)) {
                    sidechain_peaks.push((c_stats.mean, h_stats.mean));
                }
            }

            residue_shifts.push((h_backbone, n_backbone, ha_shift));
            hsqc_15n.push(UnlabeledPeak::hsqc_15n(n_backbone, h_backbone, 1.0));
            hsqc_13c.push(UnlabeledPeak::hsqc_13c(ca_shift, ha_shift, 1.0));
            for (c_shift, h_shift) in &sidechain_peaks {
                hsqc_13c.push(UnlabeledPeak::hsqc_13c(*c_shift, *h_shift, 0.8));
            }

            tocsy.push(UnlabeledPeak::tocsy(h_backbone, ha_shift, 1.0));
            tocsy.push(UnlabeledPeak::tocsy(ha_shift, h_backbone, 1.0));
            for (_, h_shift) in &sidechain_peaks {
                tocsy.push(UnlabeledPeak::tocsy(h_backbone, *h_shift, 0.8));
                tocsy.push(UnlabeledPeak::tocsy(*h_shift, h_backbone, 0.8));
            }

            ground_truth.push((seq_code, aa_name, h_backbone, n_backbone));
        }

        // NOESY: sequential dαN
        for i in 1..residues.len() {
            if residues[i] == 'P' || residues[i-1] == 'P' { continue; }
            let (h_curr, _, _) = residue_shifts[i];
            let (_, _, ha_prev) = residue_shifts[i-1];
            if h_curr > 0.0 && ha_prev > 0.0 {
                noesy.push(UnlabeledPeak::noesy(h_curr, ha_prev, 0.5));
                noesy.push(UnlabeledPeak::noesy(ha_prev, h_curr, 0.5));
            }
        }

        (hsqc_15n, hsqc_13c, tocsy, noesy, ground_truth)
    }

    /// Helper to run unified assignment and calculate accuracy.
    fn run_unified_and_score(
        hsqc_15n: &[UnlabeledPeak],
        hsqc_13c: &[UnlabeledPeak],
        tocsy: &[UnlabeledPeak],
        noesy: &[UnlabeledPeak],
        sequence: &str,
        ground_truth: &[(i32, String, f64, f64)],
    ) -> (f64, usize, usize) {
        let mut params = UnifiedAssignmentParams::default();
        params.max_iterations = 50;  // Faster for testing

        // Empty vectors for experiments not used in this test
        let empty: Vec<UnlabeledPeak> = vec![];
        let results = run_unified_assignment(
            hsqc_15n, hsqc_13c, tocsy, noesy,
            &empty, &empty,  // HSQC-TOCSY 2D
            &empty, &empty,  // HSQC-TOCSY 3D
            &empty, &empty, &empty, &empty, &empty, &empty,  // 3D triple-resonance: hnco, hncaco, hnca, hncacb, cbcaconh, hbhaconh
            sequence, &params
        );

        let backbone_results: Vec<_> = results.iter()
            .filter(|r| r.peak_type == PeakType::Backbone)
            .collect();

        let gt_by_h: std::collections::HashMap<i32, &(i32, String, f64, f64)> = ground_truth.iter()
            .map(|gt| ((gt.2 * 1000.0).round() as i32, gt))
            .collect();

        let mut correct = 0;
        let mut total = 0;

        for result in &backbone_results {
            let peak = hsqc_15n.iter().find(|p| p.id == result.peak_id);
            if let Some(peak) = peak {
                let h_shift = peak.position_ppm[1];
                let h_key = (h_shift * 1000.0).round() as i32;

                let gt = gt_by_h.iter()
                    .filter(|(&k, _)| (k - h_key).abs() < 100)
                    .min_by_key(|(&k, _)| (k - h_key).abs())
                    .map(|(_, gt)| *gt);

                if let Some((gt_seq, _, _, _)) = gt {
                    if *gt_seq == result.assigned_residue {
                        correct += 1;
                    }
                    total += 1;
                }
            }
        }

        let accuracy = if total > 0 { correct as f64 / total as f64 } else { 0.0 };
        (accuracy, correct, total)
    }

    // ==================== UNIFIED ASSIGNMENT TESTS ====================

    #[test]
    fn test_unified_simple_dipeptide() {
        let sequence = "AC";
        let (hsqc_15n, hsqc_13c, tocsy, noesy, ground_truth) = generate_idealized_peaks(sequence);

        let (accuracy, correct, total) = run_unified_and_score(
            &hsqc_15n, &hsqc_13c, &tocsy, &noesy, sequence, &ground_truth
        );

        println!("Dipeptide {}: {}/{} = {:.1}%", sequence, correct, total, accuracy * 100.0);
        assert!(accuracy >= 0.5, "Dipeptide accuracy too low: {:.1}%", accuracy * 100.0);
    }

    #[test]
    fn test_unified_short_peptide() {
        // Short peptide with chemically distinct amino acids
        let sequence = "GASWV";
        let (hsqc_15n, hsqc_13c, tocsy, noesy, ground_truth) = generate_idealized_peaks(sequence);

        println!("\n=== UNIFIED: Short Peptide {} ===", sequence);
        println!("Input: {} 15N-HSQC, {} 13C-HSQC, {} TOCSY, {} NOESY",
            hsqc_15n.len(), hsqc_13c.len(), tocsy.len(), noesy.len());

        let (accuracy, correct, total) = run_unified_and_score(
            &hsqc_15n, &hsqc_13c, &tocsy, &noesy, sequence, &ground_truth
        );

        println!("Result: {}/{} = {:.1}%", correct, total, accuracy * 100.0);
        assert!(accuracy >= 0.80, "Short peptide accuracy {:.1}% below 80%", accuracy * 100.0);
    }

    #[test]
    fn test_unified_all_amino_acids() {
        // THE MAIN TEST: All 19 amino acids (excluding Proline)
        let sequence = "ACDEFGHIKLMNQRSTVWY";
        let (hsqc_15n, hsqc_13c, tocsy, noesy, ground_truth) = generate_idealized_peaks(sequence);

        println!("\n================================================");
        println!("=== UNIFIED: ALL 19 AMINO ACIDS ===");
        println!("================================================");
        println!("Sequence: {}", sequence);
        println!("Input: {} 15N-HSQC, {} 13C-HSQC, {} TOCSY, {} NOESY",
            hsqc_15n.len(), hsqc_13c.len(), tocsy.len(), noesy.len());

        let (accuracy, correct, total) = run_unified_and_score(
            &hsqc_15n, &hsqc_13c, &tocsy, &noesy, sequence, &ground_truth
        );

        println!("\n================================================");
        println!("ACCURACY: {}/{} = {:.1}%", correct, total, accuracy * 100.0);
        println!("================================================");

        assert!(total >= 15, "Should assign at least 15 peaks, got {}", total);
        assert!(accuracy >= 0.85, "Accuracy {:.1}% below 85% target", accuracy * 100.0);
    }

    #[test]
    fn test_unified_realistic_protons() {
        // Test with realistic BMRB proton shifts (overlapping HA/HB)
        let sequence = "ACDEFGHIKLMNQRSTVWY";
        let (hsqc_15n, hsqc_13c, tocsy, noesy, ground_truth) = generate_realistic_bmrb_peaks(sequence);

        println!("\n====================================================");
        println!("=== UNIFIED: REALISTIC BMRB PROTONS (OVERLAPPING) ===");
        println!("====================================================");
        println!("Sequence: {}", sequence);
        println!("Input: {} 15N-HSQC, {} 13C-HSQC, {} TOCSY, {} NOESY",
            hsqc_15n.len(), hsqc_13c.len(), tocsy.len(), noesy.len());

        let (accuracy, correct, total) = run_unified_and_score(
            &hsqc_15n, &hsqc_13c, &tocsy, &noesy, sequence, &ground_truth
        );

        println!("\n====================================================");
        println!("ACCURACY: {}/{} = {:.1}%", correct, total, accuracy * 100.0);
        println!("====================================================");

        assert!(total >= 15, "Should assign at least 15 peaks, got {}", total);
        assert!(accuracy >= 0.55, "Accuracy {:.1}% below 55% target", accuracy * 100.0);
    }

    // ==================== DIAGNOSTIC TESTS ====================

    #[test]
    fn test_kde_scoring_quality() {
        // Verify KDE scoring produces reasonable values
        use crate::inference::scoring::KDEScorer;
        use crate::inference::scoring::ShiftScorer;

        let scorer = KDEScorer::new();

        // ALA CB should score high for ~19 ppm
        let ala_cb_score = scorer.score("ALA", "CB", 19.0);
        assert!(ala_cb_score > 0.1, "ALA CB at 19 ppm should score well: {}", ala_cb_score);

        // ALA CB should score low for ~60 ppm (wrong region)
        let ala_cb_wrong = scorer.score("ALA", "CB", 60.0);
        assert!(ala_cb_wrong < ala_cb_score, "ALA CB at 60 ppm should score lower");

        // SER CB at ~64 ppm (distinctive)
        let ser_cb_score = scorer.score("SER", "CB", 64.0);
        assert!(ser_cb_score > 0.1, "SER CB at 64 ppm should score well: {}", ser_cb_score);

        println!("KDE scoring test passed:");
        println!("  ALA CB @19ppm: {:.3}", ala_cb_score);
        println!("  ALA CB @60ppm: {:.3}", ala_cb_wrong);
        println!("  SER CB @64ppm: {:.3}", ser_cb_score);
    }

    #[test]
    fn test_proton_collision_awareness() {
        // Diagnostic: show proton collisions that the unified approach handles
        let kde = KDEDatabase::load_embedded();

        println!("\n=== Proton Mode Collisions (within 0.01 ppm) ===");
        let all_aas = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
                       "ILE", "LEU", "LYS", "MET", "PHE", "SER", "THR", "TRP", "TYR", "VAL"];

        let mut all_protons: Vec<(&str, &str, f64)> = Vec::new();
        for aa in &all_aas {
            for proton in get_spin_system_protons(aa) {
                if let Some(grid) = kde.get(aa, proton) {
                    all_protons.push((aa, proton, grid.mode()));
                }
            }
        }

        let mut collision_count = 0;
        for i in 0..all_protons.len() {
            for j in (i+1)..all_protons.len() {
                let (aa1, p1, h1) = all_protons[i];
                let (aa2, p2, h2) = all_protons[j];
                if aa1 != aa2 && (h1 - h2).abs() < 0.01 {
                    println!("  {} {} ({:.3}) <-> {} {} ({:.3}) = {:.3} ppm",
                        aa1, p1, h1, aa2, p2, h2, (h1-h2).abs());
                    collision_count += 1;
                }
            }
        }

        println!("\nTotal collisions within 0.01 ppm: {}", collision_count);
        println!("The unified approach handles these via carbon typing (weight 6.0)");
    }

    #[test]
    fn test_carbon_discrimination() {
        // Diagnostic: show how carbon shifts discriminate amino acids
        let kde = KDEDatabase::load_embedded();

        println!("\n=== Carbon Discrimination by Amino Acid ===");

        let groups = [
            ("Distinct CB", vec![("GLY", "no CB"), ("ALA", "CB~19"), ("SER", "CB~64")]),
            ("Branched", vec![("VAL", "CB~32"), ("ILE", "CB~38"), ("LEU", "CB~42"), ("THR", "CB~69")]),
        ];

        for (group_name, aas) in &groups {
            println!("\n{}:", group_name);
            for (aa, desc) in aas {
                if let Some(cb) = kde.get(aa, "CB") {
                    println!("  {} CB mode={:.1} ppm ({})", aa, cb.mode(), desc);
                } else {
                    println!("  {} {} (no CB)", aa, desc);
                }
            }
        }
    }

    // ==================== VERBOSE TEST ====================

    #[test]
    #[ignore]  // Run with: cargo test test_verbose -- --ignored --nocapture
    fn test_verbose_short_peptide() {
        // Run the unified assignment with verbose mode to see what's happening
        let sequence = "GASWV";
        let (hsqc_15n, hsqc_13c, tocsy, noesy, ground_truth) = generate_idealized_peaks(sequence);

        println!("\n");
        println!("***************************************************************");
        println!("*  VERBOSE TEST: Running unified assignment with full output  *");
        println!("***************************************************************");
        println!("\n");

        let mut params = UnifiedAssignmentParams::default();
        params.verbose = true;  // Enable verbose output
        params.max_iterations = 30;  // Fewer iterations for readable output

        // Empty vectors for experiments not used in this test
        let empty: Vec<UnlabeledPeak> = vec![];
        let results = run_unified_assignment(
            &hsqc_15n, &hsqc_13c, &tocsy, &noesy,
            &empty, &empty,  // HSQC-TOCSY 2D
            &empty, &empty,  // HSQC-TOCSY 3D
            &empty, &empty, &empty, &empty, &empty, &empty,  // 3D triple-resonance: hnco, hncaco, hnca, hncacb, cbcaconh, hbhaconh
            sequence, &params
        );

        // Score the results
        let backbone_results: Vec<_> = results.iter()
            .filter(|r| r.peak_type == PeakType::Backbone)
            .collect();

        let gt_by_h: std::collections::HashMap<i32, &(i32, String, f64, f64)> = ground_truth.iter()
            .map(|gt| ((gt.2 * 1000.0).round() as i32, gt))
            .collect();

        let mut correct = 0;
        let mut total = 0;

        for result in &backbone_results {
            let peak = hsqc_15n.iter().find(|p| p.id == result.peak_id);
            if let Some(peak) = peak {
                let h_shift = peak.position_ppm[1];
                let h_key = (h_shift * 1000.0).round() as i32;

                let gt = gt_by_h.iter()
                    .filter(|(&k, _)| (k - h_key).abs() < 100)
                    .min_by_key(|(&k, _)| (k - h_key).abs())
                    .map(|(_, gt)| *gt);

                if let Some((gt_seq, gt_name, _, _)) = gt {
                    let is_correct = *gt_seq == result.assigned_residue;
                    if is_correct { correct += 1; }
                    total += 1;

                    let status = if is_correct { "✓" } else { "✗" };
                    println!("{} H={:.3}: assigned={}, expected={} ({})",
                        status, h_shift, result.assigned_residue, gt_seq, gt_name);
                }
            }
        }

        let accuracy = if total > 0 { correct as f64 / total as f64 } else { 0.0 };
        println!("\nFINAL ACCURACY: {}/{} = {:.1}%", correct, total, accuracy * 100.0);
    }
}
