---
name: ppi-interface
description: Analyze protein-protein interfaces — BSA, alanine scanning, hotspot residues, ColabFold prediction, FreeBindCraft binder design
triggers:
  - "analyze interface between chain A and chain B"
  - "predict complex structure from two sequences"
  - "find hotspot residues at this protein interface"
  - "alanine scanning on this antibody-antigen complex"
  - "design a binder against [target protein]"
  - "which residues drive binding at this PPI?"
inputs:
  - PDB file path / 4-letter PDB ID
  - Or "predict:SEQA:SEQB" for ColabFold Mode B
outputs:
  - interface_residues.csv — all interface residues with BSA & hotspot scores
  - hotspots.json — hotspot list for FreeBindCraft
  - interface_bsa.png — BSA bar chart
  - contact_map.png — inter-chain contact heatmap
  - composition_radar.png — polar/apolar composition radar
  - interface_report.txt — full text report
modes:
  - mode_a: "Interface analysis from PDB file (CPU, fast)"
  - mode_b: "ColabFold prediction from sequences (GPU recommended)"
  - mode_c: "FreeBindCraft de novo binder design (GPU required)"
---

# Protein-Protein Interface Analysis & Hotspot Prediction Skill

## Modes

### Mode A — Interface Analysis (CPU, ~2–10 min)
Given a PDB file (experimental or predicted):
1. Interface residues via BSA differential
2. Contact map (Cα–Cα < 8Å, heavy atom < 5Å)
3. Computational alanine scanning — SASA-proxy for ΔΔG
4. Hotspot ranking (BSA ≥ 25 Å² = predicted hotspot)
5. H-bond and salt bridge detection
6. Shape complementarity proxy

### Mode B — Complex Prediction (GPU recommended, ~5–30 min)
Given two amino acid sequences, predict the heterodimer using ColabFold AlphaFold2-Multimer v3, then automatically run Mode A on the top-ranked model.

### Mode C — De Novo Binder Design (GPU required)
FreeBindCraft hallucination of novel binders against target. See `ppi_pipeline.py` for full implementation.

## Usage

```python
from ppi_pipeline import run_pipeline

# Mode A: Analyze a PDB file or ID
results = run_pipeline(
    input_source="4ZQK",   # PDB ID, file path, or "predict:seqA:seqB"
    chain_a="A",
    chain_b="B",
    out_dir="results_pdl1",
)

# Mode B: Predict complex from sequences
results = run_pipeline(
    input_source="predict",
    chain_a="A", chain_b="B",
    out_dir="results_predicted",
    run_colabfold=True,
    seq_a="<nanobody_sequence>",
    seq_b="<antigen_sequence>",
)
```

## Scientific Basis
- Hotspot definition: ΔΔG_bind ≥ 2.0 kcal/mol upon Ala mutation (Bogan & Thorn 1998)
- SASA-based proxy correlates ~0.6 with experimental alanine scanning (Moreira et al. 2007)
- Shape complementarity: Sc > 0.65 = well-packed interface (Lawrence & Colman 1993)
- ipTM > 0.75 = high-confidence interface prediction (ColabFold)

## Demo: PD-1/PD-L1 (PDB 4ZQK)
```bash
python ppi_pipeline.py
```
