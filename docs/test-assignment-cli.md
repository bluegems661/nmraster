# Test Assignment CLI

A command-line tool for testing the unified NMR assignment algorithm against ground truth data.

## Quick Start

```bash
# Build and run (from project root)
cd src-tauri
cargo run --bin test_assignment -- synthetic --sequence THFG
```

## Usage

```
test_assignment [OPTIONS] <COMMAND>

Commands:
  synthetic  Generate synthetic data from a sequence
  bmrb       Fetch real data from a BMRB entry

Global Options:
  -v, --verbose                    Enable verbose output during belief propagation
      --exclude <EXPERIMENTS>      Exclude specific experiment types (comma-separated)
      --only <EXPERIMENTS>         Only use specific experiment types (comma-separated)
  -h, --help                       Print help
  -V, --version                    Print version

Experiment Types (for --exclude/--only):
  15n-hsqc, 13c-hsqc, tocsy, noesy, hsqc-tocsy-15n, hsqc-tocsy-13c
```

## Modes

### Synthetic Mode

Generate "perfect" peaks from KDE statistical modes for a given amino acid sequence.

```bash
# Basic usage - perfect peaks at KDE mode positions
cargo run --bin test_assignment -- synthetic --sequence THFGKMIVW

# With noise - adds Gaussian variation (0.1 = 10% of typical std)
cargo run --bin test_assignment -- synthetic --sequence THFGKMIVW --noise 0.1

# Short sequence for quick testing
cargo run --bin test_assignment -- synthetic --sequence AC

# Verbose - see belief propagation iterations
cargo run --bin test_assignment -- -v synthetic --sequence THFG
```

**Options:**
- `-s, --sequence <SEQ>` - One-letter amino acid sequence (required)
- `-n, --noise <FLOAT>` - Noise level as fraction of typical std dev (default: 0.0)

### BMRB Mode

Fetch real deposited chemical shifts from the BioMagResBank (BMRB) database.

```bash
# Fetch all residues from an entry
cargo run --bin test_assignment -- bmrb --entry 4493

# Fetch specific residue range (1-indexed, inclusive)
cargo run --bin test_assignment -- bmrb --entry 4493 --residues 1-10
cargo run --bin test_assignment -- bmrb --entry 4493 --residues 45-62

# Verbose output
cargo run --bin test_assignment -- -v bmrb --entry 4493 --residues 1-5
```

**Options:**
- `-e, --entry <ID>` - BMRB entry ID number (required)
- `-r, --residues <RANGE>` - Residue range like "1-10" (optional, defaults to all)

**Finding BMRB Entries:**
- Browse entries at https://bmrb.io/
- Example entries with good chemical shift coverage:
  - 4493 - Ubiquitin mutant (76 residues)
  - 15000 - GB1 domain (56 residues)
  - 6457 - Calmodulin (148 residues)

## Output

The tool produces a results table and summary statistics:

```
===============================================================================
                        UNIFIED ASSIGNMENT TEST RESULTS
===============================================================================

Sequence: THFG (4 residues)
Mode: Synthetic (perfect)

-------------------------------------------------------------------------------
 Peak ID  | Type | Predicted | Actual | Match | Confidence | Atom
-------------------------------------------------------------------------------
 a1b2c3.. | BB   | 2         | 2      |   OK  | 94.2%      | H/N
 d4e5f6.. | BB   | 3         | 3      |   OK  | 87.5%      | H/N
 g7h8i9.. | C    | 1         | 1      |   OK  | 91.3%      | CA/HA
 j0k1l2.. | C    | 2         | 3      | MISS  | 45.2%      | CB/HB
-------------------------------------------------------------------------------

SUMMARY:
  Total peaks:       12
  Correct:           10
  Overall accuracy:  83.3%

  Backbone peaks:    3
  Backbone correct:  3
  Backbone accuracy: 100.0%
```

**Columns:**
- **Peak ID** - First 8 characters of UUID
- **Type** - BB (backbone 15N-HSQC) or C (carbon 13C-HSQC)
- **Predicted** - Residue number assigned by algorithm
- **Actual** - Ground truth residue number
- **Match** - OK or MISS
- **Confidence** - Algorithm's confidence in assignment (0-100%)
- **Atom** - Atom pair (e.g., H/N for backbone, CA/HA for alpha carbon)

## Verbose Mode

With `-v` flag, the tool shows belief propagation progress:

```
┌─── Iteration 50 (50% - REFINE) ───────────────────────────────
│ Parameters:
│   H tolerance: 0.0350 ppm
│   C tolerance: 2.00 ppm
│   TOCSY weight: 0.90
│   Typing weight: 4.50
│   Seq-type weight: 5.00
│ Max belief change: 0.012345
│
│ Current backbone peak beliefs (showing top 3):
│   Peak  0 (H=8.21, N=120.1): pos 2 (THR) 87.5%
│   Peak  1 (H=8.45, N=118.3): pos 3 (HIS) 92.1%
└───────────────────────────────────────────────────────────────
```

## Generated Peak Types

The tool generates eleven types of NMR peaks from the input data:

### 2D Experiments

1. **15N-HSQC** - Backbone H-N correlations (one per non-proline residue)
2. **13C-HSQC** - C-H correlations (CA-HA, CB-HB, sidechain carbons)
3. **TOCSY** - H-H correlations within each residue spin system
4. **NOESY** - Sequential H(i) to HA(i-1) correlations
5. **15N-HSQC-TOCSY** - Nitrogen-anchored TOCSY (N, H_tocsy) - links backbone NH to all protons in spin system
6. **13C-HSQC-TOCSY** - Carbon-anchored TOCSY (C, H_tocsy) - links each carbon to all protons in spin system

### 3D Triple-Resonance Experiments

7. **HNCO** - (H, N, CO) - Correlates NH(i) with CO(i-1) carbonyl carbon
8. **HNCA** - (H, N, CA) - Correlates NH(i) with CA(i) (strong) and CA(i-1) (weak, ~30% intensity)
9. **HNCACB** - (H, N, CA/CB) - CA/CB(i) have positive intensity, CA/CB(i-1) have negative intensity
10. **CBCACONH** - (H, N, CA/CB) - Only shows i-1 carbons (CA and CB from previous residue)
11. **HBHACONH** - (H, N, HA/HB) - Only shows i-1 aliphatic protons

### HSQC-TOCSY Experiments

HSQC-TOCSY experiments combine the resolution of HSQC with the spin system connectivity of TOCSY:

- **15N-HSQC-TOCSY**: Each peak correlates a backbone nitrogen with a proton in the same residue. This anchors the spin system to the backbone NH.
- **13C-HSQC-TOCSY**: Each peak correlates a carbon with a proton in the same residue. This provides heavy-atom-anchored spin system information.

These experiments are particularly valuable because:
- They provide higher confidence spin system grouping than regular TOCSY (heavy-atom anchoring reduces ambiguity)
- The unified assignment algorithm weights HSQC-TOCSY correlations at **1.5x** the regular TOCSY weight

### Triple-Resonance Sequential Assignment

The 3D triple-resonance experiments provide unambiguous sequential connectivity:

1. **HNCA** shows CA(i) strong and CA(i-1) weak at each backbone NH
2. **CBCACONH** shows only CA/CB(i-1) at each backbone NH
3. **Matching**: When CA(i) from HNCA at NH(n) matches CA(i-1) from CBCACONH at NH(n+1), residue n precedes residue n+1

This CA/CB chemical shift matching provides definitive backbone sequential assignment, independent of NOE-based methods.

**Note**: 3D peak generation is implemented, but the integration with the belief propagation assignment algorithm is pending.

## Interpreting Results

| Accuracy | Interpretation |
|----------|----------------|
| 90-100%  | Excellent - algorithm working well |
| 70-90%   | Good - some ambiguity in similar residues |
| 50-70%   | Moderate - overlapping chemical shifts |
| <50%     | Poor - algorithm needs improvement or data quality issues |

**Synthetic vs BMRB:**
- Synthetic mode with noise=0 tests the algorithm under ideal conditions
- BMRB mode tests with real chemical shift variability
- Expect 10-30% lower accuracy on BMRB data vs synthetic

## Examples

```bash
# Quick sanity check
cargo run --bin test_assignment -- synthetic --sequence AC

# Test with realistic sequence
cargo run --bin test_assignment -- synthetic --sequence MQIFVKTLTGK --noise 0.05

# Real data validation
cargo run --bin test_assignment -- bmrb --entry 4493 --residues 1-20

# Debug mode - see what's happening
cargo run --bin test_assignment -- -v synthetic --sequence THFG
```

## Experiment Filtering

Use `--exclude` or `--only` to control which experiment types are included in the test.

### Exclude Experiments

Exclude specific experiments while keeping all others:

```bash
# Run without HSQC-TOCSY experiments
cargo run --bin test_assignment -- synthetic --sequence ACDEF --exclude hsqc-tocsy-15n,hsqc-tocsy-13c

# Run without NOESY (test TOCSY-only grouping)
cargo run --bin test_assignment -- synthetic --sequence ACDEF --exclude noesy

# Run without any TOCSY variants (test typing-only)
cargo run --bin test_assignment -- synthetic --sequence ACDEF --exclude tocsy,hsqc-tocsy-15n,hsqc-tocsy-13c
```

### Only Use Specific Experiments

Include only the specified experiments:

```bash
# Backbone-only analysis (15N-HSQC + TOCSY)
cargo run --bin test_assignment -- synthetic --sequence ACDEF --only 15n-hsqc,tocsy

# Test with only HSQC experiments (no connectivity)
cargo run --bin test_assignment -- synthetic --sequence ACDEF --only 15n-hsqc,13c-hsqc

# Full analysis with all experiments
cargo run --bin test_assignment -- synthetic --sequence ACDEF
# (equivalent to no filter)
```

### Experiment Type Names

| Name | Description |
|------|-------------|
| `15n-hsqc` | Backbone 15N-HSQC (H-N correlations) |
| `13c-hsqc` | Aliphatic 13C-HSQC (C-H correlations) |
| `tocsy` | 2D TOCSY (H-H within spin system) |
| `noesy` | 2D NOESY (sequential H-HA correlations) |
| `hsqc-tocsy-15n` | 15N-HSQC-TOCSY (N-anchored spin system) |
| `hsqc-tocsy-13c` | 13C-HSQC-TOCSY (C-anchored spin system) |
| `hnco` | 3D HNCO (H, N, CO of i-1) |
| `hnca` | 3D HNCA (H, N, CA of i and i-1) |
| `hncacb` | 3D HNCACB (H, N, CA/CB with sign encoding) |
| `cbcaconh` | 3D CBCACONH (H, N, CA/CB of i-1 only) |
| `hbhaconh` | 3D HBHACONH (H, N, HA/HB of i-1 only) |

### Use Cases

**Ablation studies** - Test how each experiment type contributes to accuracy:
```bash
# Baseline with all experiments
cargo run --bin test_assignment -- synthetic --sequence THFGKMIVW

# Without HSQC-TOCSY (measure contribution of heavy-atom anchoring)
cargo run --bin test_assignment -- synthetic --sequence THFGKMIVW --exclude hsqc-tocsy-15n,hsqc-tocsy-13c

# Without NOESY (measure contribution of sequential connectivity)
cargo run --bin test_assignment -- synthetic --sequence THFGKMIVW --exclude noesy
```

**Simulating limited data** - Test algorithm robustness:
```bash
# Simulate having only backbone data
cargo run --bin test_assignment -- synthetic --sequence THFGKMIVW --only 15n-hsqc,tocsy,noesy
```

**Testing 3D triple-resonance experiments**:
```bash
# Include 3D experiments in test (note: peak generation only, BP integration pending)
cargo run --bin test_assignment -- bmrb --entry 4493 --residues 1-10

# Exclude 3D experiments to compare (baseline 2D-only)
cargo run --bin test_assignment -- bmrb --entry 4493 --residues 1-10 --exclude hnco,hnca,hncacb,cbcaconh,hbhaconh

# Test with only HNCA and HNCACB (most diagnostic for sequential assignment)
cargo run --bin test_assignment -- synthetic --sequence THFGKMIVW --only 15n-hsqc,hnca,hncacb
```
