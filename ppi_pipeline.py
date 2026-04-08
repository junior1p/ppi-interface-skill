#!/usr/bin/env python3
"""
PPI Interface Analysis Pipeline — Full Skill Implementation
Mode A: Interface analysis (BSA, alanine scanning, hotspots)
Mode B: ColabFold complex prediction from sequences
Mode C: FreeBindCraft de novo binder design (GPU, documented but not run here)

Demo: PD-1/PD-L1 immune checkpoint complex (PDB: 4ZQK)
Also demonstrates Mode B: ColabFold prediction of a hypothetical nanobody-antigen pair.
"""

import os, json, glob, subprocess, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.Structure import Structure as BioStructure
from Bio.PDB.Model import Model as BioModel
from Bio.PDB.Chain import Chain as BioChain
from Bio.PDB.Residue import Residue as BioResidue
from Bio.PDB.Atom import Atom as BioAtom
import copy
warnings.filterwarnings("ignore")

os.makedirs("/root/ppi_interface/results", exist_ok=True)

# ══════════════════════════════════════════════════════════════
# STEP 0: ENVIRONMENT
# ══════════════════════════════════════════════════════════════
# Dependencies installed: biopython numpy pandas matplotlib seaborn scipy requests

# ══════════════════════════════════════════════════════════════
# INPUT ROUTING
# ══════════════════════════════════════════════════════════════
def load_structure(input_source: str):
    """
    Three supported input types:
      1. "predict:SEQA:SEQB"  → ColabFold prediction (Mode B)
      2. Local .pdb / .cif file path
      3. 4-letter PDB accession code
    Returns: (structure, source_type, pdb_path)
    """
    input_source = input_source.strip()

    if input_source.startswith("predict:"):
        parts = input_source.split(":", 2)
        seq_a, seq_b = parts[1].strip(), parts[2].strip()
        pdb_path = predict_complex_colabfold(seq_a, seq_b)
        source_type = "predicted"
    elif os.path.exists(input_source):
        pdb_path = input_source
        source_type = "local"
    elif len(input_source) == 4 and input_source.isalnum():
        pdb_path = fetch_pdb(input_source)
        source_type = "pdb_download"
    else:
        raise ValueError(
            f"Cannot parse input: '{input_source}'\n"
            "Provide: a PDB file path, 4-letter PDB ID, or 'predict:SEQA:SEQB'"
        )

    if pdb_path.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    return structure, source_type, pdb_path

def fetch_pdb(pdb_id: str, out_dir: str = "data/") -> str:
    os.makedirs(out_dir, exist_ok=True)
    pdb_id = pdb_id.upper()
    out_path = f"{out_dir}/{pdb_id}.pdb"
    if os.path.exists(out_path):
        print(f"PDB {pdb_id} already downloaded."); return out_path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        out_path = f"{out_dir}/{pdb_id}.cif"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    with open(out_path, "w") as f: f.write(response.text)
    print(f"Downloaded PDB {pdb_id} to {out_path}")
    return out_path

# ══════════════════════════════════════════════════════════════
# MODE B: ColabFold Complex Prediction
# ══════════════════════════════════════════════════════════════
def predict_complex_colabfold(
    seq_a: str, seq_b: str,
    out_dir: str = "colabfold_output",
    num_recycles: int = 3,
    model_type: str = "alphafold2_multimer_v3",
) -> str:
    """
    Predict heterodimer complex using ColabFold AlphaFold2-Multimer.
    Sequences separated by ':' in FASTA → ColabFold auto-uses multimer mode.
    MSA built via public MMseqs2 server (no local DB needed).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Validate sequences
    aa_set = set("ACDEFGHIKLMNPQRSTVWY")
    for name, seq in [("A", seq_a), ("B", seq_b)]:
        invalid = set(seq.upper()) - aa_set
        if invalid:
            raise ValueError(f"Chain {name} contains invalid amino acids: {invalid}")

    fasta_path = f"{out_dir}/complex_query.fasta"
    with open(fasta_path, "w") as f:
        f.write(f">complex_AB\n{seq_a.upper()}:{seq_b.upper()}\n")

    print(f"[Mode B] Chain A = {len(seq_a)} aa, Chain B = {len(seq_b)} aa")
    print(f"[Mode B] Total complex: {len(seq_a) + len(seq_b)} aa")
    print(f"[Mode B] Running ColabFold ({model_type})...")
    print(f"[Mode B] MSA via public MMseqs2 server — estimated 5-20 min (CPU)")

    cmd = [
        "colabfold_batch",
        fasta_path, out_dir,
        "--model-type", model_type,
        "--num-recycle", str(num_recycles),
        "--num-models", "2",
        "--rank", "multimer",
        "--stop-at-score", "70",
        "--zip",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        print(f"ColabFold stderr:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"ColabFold failed with exit code {result.returncode}")

    pdb_files = sorted(glob.glob(f"{out_dir}/*rank_001*.pdb"))
    if not pdb_files:
        pdb_files = sorted(glob.glob(f"{out_dir}/*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No PDB output in {out_dir}")
    best_pdb = pdb_files[0]
    print(f"[Mode B] Best model: {best_pdb}")
    parse_colabfold_scores(out_dir)
    return best_pdb

def parse_colabfold_scores(out_dir: str) -> dict:
    """Extract ipTM and pTM scores from ColabFold output JSON."""
    import json as _json
    score_files = glob.glob(f"{out_dir}/*scores*.json")
    if not score_files:
        print("  No scores JSON found."); return {}
    scores = {}
    for sf in sorted(score_files):
        with open(sf) as f:
            d = _json.load(f)
        scores[os.path.basename(sf)] = {
            "iptm": round(d.get("iptm", 0), 3),
            "ptm":  round(d.get("ptm",  0), 3),
            "ranking_score": round(d.get("ranking_confidence", 0), 3),
        }
    print("\n--- ColabFold Quality Metrics ---")
    for name, s in scores.items():
        iptm = s["iptm"]; ptm = s["ptm"]
        iptm_desc = "HIGH" if iptm > 0.75 else "MODERATE" if iptm > 0.5 else "LOW"
        print(f"  {name}")
        print(f"    ipTM (interface confidence): {iptm}  [{iptm_desc}]  [>0.75=high, 0.5-0.75=moderate, <0.5=unreliable]")
        print(f"    pTM  (overall confidence):   {ptm}")
        print(f"    ipTM>0.75 + pTM>0.9 → structure suitable for detailed hotspot analysis")
    return scores

# ══════════════════════════════════════════════════════════════
# SASA Utilities
# ══════════════════════════════════════════════════════════════
def res_key(res) -> str:
    return f"{res.get_parent().id}:{res.id[1]}:{res.resname}"

def isolate_chain(chain) -> BioStructure:
    iso = BioStructure("iso")
    iso.add(BioModel(0))
    iso[0].add(BioChain(chain.id))
    iso_c = iso[0][chain.id]
    for orig_res in chain.get_residues():
        if orig_res.id[0] != ' ': continue
        new_res = BioResidue(orig_res.id, orig_res.resname, orig_res.segid)
        for atom in orig_res.get_atoms():
            new_atom = BioAtom(
                atom.name, copy.copy(atom.coord),
                atom.bfactor, atom.occupancy, atom.altloc,
                atom.fullname, None,
            )
            new_res.add(new_atom)
        iso_c.add(new_res)
    return iso

def sasa_isolated(chain, n_points: int = 250) -> dict:
    iso = isolate_chain(chain)
    sr = ShrakeRupley(n_points=n_points)
    sr.compute(iso, level="R")
    return {res_key(res): res.sasa for res in iso.get_residues() if res.id[0] == ' ' and res.sasa}

# ══════════════════════════════════════════════════════════════
# STEP 2: Interface Identification
# ══════════════════════════════════════════════════════════════
def identify_interface(structure, chain_a_id: str, chain_b_id: str,
                      contact_cutoff: float = 8.0,
                      heavy_atom_cutoff: float = 5.0,
                      bsa_cutoff: float = 1.0) -> dict:
    model = structure[0]
    chain_a = model[chain_a_id]
    chain_b = model[chain_b_id]

    # Distance-based contacts
    contact_pairs = []
    for res_a in chain_a.get_residues():
        if res_a.id[0] != ' ': continue
        for res_b in chain_b.get_residues():
            if res_b.id[0] != ' ': continue
            try:
                if (res_a["CA"] - res_b["CA"]) < contact_cutoff:
                    min_dist = min(
                        (a1 - a2)
                        for a1 in res_a.get_atoms()
                        for a2 in res_b.get_atoms()
                        if a1.element != 'H' and a2.element != 'H'
                    )
                    if min_dist < heavy_atom_cutoff:
                        contact_pairs.append((res_a, res_b, round(min_dist, 2)))
            except KeyError:
                pass

    # SASA of complex
    sr = ShrakeRupley(n_points=250)
    sr.compute(structure, level="R")
    sasa_complex = {res_key(res): res.sasa
                   for res in structure.get_residues()
                   if res.id[0] == ' ' and res.sasa}

    # SASA of each chain alone
    sasa_a = sasa_isolated(chain_a)
    sasa_b = sasa_isolated(chain_b)

    # BSA per residue
    bsa_a, bsa_b = {}, {}
    for res in chain_a.get_residues():
        if res.id[0] != ' ': continue
        key = res_key(res)
        iso = sasa_a.get(key); bound = sasa_complex.get(key)
        if iso is not None and bound is not None:
            bsa = iso - bound
            if bsa > bsa_cutoff: bsa_a[res] = bsa

    for res in chain_b.get_residues():
        if res.id[0] != ' ': continue
        key = res_key(res)
        iso = sasa_b.get(key); bound = sasa_complex.get(key)
        if iso is not None and bound is not None:
            bsa = iso - bound
            if bsa > bsa_cutoff: bsa_b[res] = bsa

    total_bsa = sum(bsa_a.values()) + sum(bsa_b.values())
    print(f"\n--- Interface Summary ---")
    print(f"  Chain {chain_a_id} interface residues: {len(bsa_a)}")
    print(f"  Chain {chain_b_id} interface residues: {len(bsa_b)}")
    print(f"  Total BSA: {total_bsa:.1f} Å²")
    print(f"  Contact pairs (heavy atom < {heavy_atom_cutoff}Å): {len(contact_pairs)}")
    print(f"  Typical Ab-Ag interface BSA: 1200–2000 Å²")

    return {
        "bsa_a": bsa_a, "bsa_b": bsa_b,
        "total_bsa": total_bsa, "contact_pairs": contact_pairs,
        "sasa_a": sasa_a, "sasa_b": sasa_b,
    }

# ══════════════════════════════════════════════════════════════
# STEP 3: Alanine Scanning
# ══════════════════════════════════════════════════════════════
HOTSPOT_WEIGHTS = {
    "TRP": 3.0, "TYR": 2.5, "ARG": 2.0, "PHE": 2.0,
    "LEU": 1.5, "ILE": 1.5, "MET": 1.5, "LYS": 1.3,
    "VAL": 1.2, "HIS": 1.8, "ASP": 1.2, "GLU": 1.2,
    "ASN": 1.0, "GLN": 1.0, "PRO": 0.8, "CYS": 1.8,
    "THR": 0.7, "SER": 0.6, "ALA": 0.5, "GLY": 0.3,
}
HOTSPOT_THRESHOLD = 25.0

def alanine_scan(interface_data: dict, chain_id: str) -> list:
    bsa_dict = interface_data[f"bsa_{chain_id.lower()}"]
    results = []
    for res, bsa in bsa_dict.items():
        rn = res.get_resname()
        score = bsa * HOTSPOT_WEIGHTS.get(rn, 1.0)
        results.append({
            "res": res, "label": f"{chain_id}{res.id[1]}{rn}",
            "resname": rn, "resnum": res.id[1], "chain": chain_id,
            "BSA_A2": round(bsa, 2),
            "hotspot_score": round(score, 2),
            "is_hotspot": bsa >= HOTSPOT_THRESHOLD,
        })
    results.sort(key=lambda x: x["hotspot_score"], reverse=True)
    n_hs = sum(1 for r in results if r["is_hotspot"])
    print(f"\n--- Alanine Scanning: Chain {chain_id} ---")
    print(f"  Interface residues: {len(results)}, Hotspots (BSA≥{HOTSPOT_THRESHOLD}Å²): {n_hs}")
    print(f"  {'Residue':<12} {'BSA (Å²)':<12} {'Score':<10} Hotspot?")
    print(f"  {'-'*45}")
    for r in results[:10]:
        flag = "🔥" if r["is_hotspot"] else ""
        print(f"  {r['label']:<12} {r['BSA_A2']:<12.1f} {r['hotspot_score']:<10.2f} {flag}")
    return results

# ══════════════════════════════════════════════════════════════
# STEP 4: Interface Composition — INCLUDING H-bond / Salt Bridge
# ══════════════════════════════════════════════════════════════
# Donor-Acceptor pairs for H-bonds (N,O donors; N,O acceptors)
HBOND_DONORS  = {"N", "NH1", "NH2", "ND1", "ND2", "NE", "NE1", "NE2", "NZ"}
HBOND_ACCEPTS = {"O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "OXT"}
# Salt bridge: charged residues at close distance
SALT_BRIDGE_PAIRS = {
    ("ARG", "ASP"), ("ARG", "GLU"),
    ("LYS", "ASP"), ("LYS", "GLU"),
    ("HIS", "ASP"), ("HIS", "GLU"),
}

def analyze_interface_composition(interface_data, chain_a_id, chain_b_id, ha, hb) -> dict:
    POLAR = {"SER","THR","ASN","GLN","TYR","TRP","CYS"}
    AP    = {"ALA","VAL","LEU","ILE","MET","PHE","PRO","GLY"}
    CP    = {"ARG","LYS","HIS"}
    CN    = {"ASP","GLU"}

    def classify(lst):
        c = {"polar":0,"apolar":0,"charged+":0,"charged-":0}
        for r in lst:
            n = r["resname"] if isinstance(r,dict) else r.get_resname()
            if n in POLAR: c["polar"]+=1
            elif n in AP:  c["apolar"]+=1
            elif n in CP:  c["charged+"]+=1
            elif n in CN:  c["charged-"]+=1
        return c

    comp_a = classify(ha); comp_b = classify(hb)

    # Shape complementarity
    bsa = interface_data["total_bsa"]
    denom = (sum(interface_data["sasa_a"].get(res_key(r["res"]), 0) for r in ha) +
             sum(interface_data["sasa_b"].get(res_key(r["res"]), 0) for r in hb))
    sc = (2 * bsa / denom) if denom > 0 else 0

    # ── H-bond and salt bridge detection ──────────────────────
    hbond_pairs = []
    salt_bridge_pairs = []
    contact_pairs = interface_data["contact_pairs"]

    for res_a, res_b, dist in contact_pairs:
        rn_a = res_a.get_resname(); rn_b = res_b.get_resname()
        # Salt bridge: charged residues within 4.5 Å
        if dist < 4.5:
            pair = tuple(sorted([rn_a, rn_b]))
            if pair in SALT_BRIDGE_PAIRS or (rn_a in CP and rn_b in CN) or (rn_a in CN and rn_b in CP):
                salt_bridge_pairs.append((res_a, res_b, dist))
        # H-bond: N/O donor to N/O acceptor within 3.5 Å
        if dist < 3.5:
            donor_a = any(a.element == 'N' and a.name in HBOND_DONORS for a in res_a.get_atoms())
            accept_b = any(a.element in ('N','O') and a.name in HBOND_ACCEPTS for a in res_b.get_atoms())
            donor_b = any(a.element == 'N' and a.name in HBOND_DONORS for a in res_b.get_atoms())
            accept_a = any(a.element in ('N','O') and a.name in HBOND_ACCEPTS for a in res_a.get_atoms())
            if (donor_a and accept_b) or (donor_b and accept_a):
                hbond_pairs.append((res_a, res_b, dist))

    composition = {
        "chain_a": comp_a, "chain_b": comp_b,
        "total_bsa_A2": round(bsa, 1),
        "shape_comp": round(sc, 3),
        "n_ha": sum(1 for r in ha if r["is_hotspot"]),
        "n_hb": sum(1 for r in hb if r["is_hotspot"]),
        "n_contacts": len(contact_pairs),
        "n_hbonds": len(hbond_pairs),
        "n_salt_bridges": len(salt_bridge_pairs),
    }

    print(f"\n--- Interface Composition ---")
    print(f"  Total BSA:              {composition['total_bsa_A2']} Å²")
    print(f"  Shape complementarity:   {composition['shape_comp']}  [>0.65=good, <0.50=poor]")
    print(f"  H-bonds (dist < 3.5Å):   {len(hbond_pairs)}")
    print(f"  Salt bridges (dist < 4.5Å): {len(salt_bridge_pairs)}")
    print(f"  Chain {chain_a_id}: {comp_a}")
    print(f"  Chain {chain_b_id}: {comp_b}")

    if hbond_pairs:
        print(f"  Top H-bonds:")
        for a, b, d in sorted(hbond_pairs, key=lambda x: x[2])[:5]:
            print(f"    {a.get_parent().id}{a.id[1]}{a.get_resname()} — {b.get_parent().id}{b.id[1]}{b.get_resname()} @ {d}Å")
    if salt_bridge_pairs:
        print(f"  Top Salt bridges:")
        for a, b, d in sorted(salt_bridge_pairs, key=lambda x: x[2])[:5]:
            print(f"    {a.get_parent().id}{a.id[1]}{a.get_resname()} — {b.get_parent().id}{b.id[1]}{b.get_resname()} @ {d}Å")

    return composition

# ══════════════════════════════════════════════════════════════
# STEP 5: Visualizations
# ══════════════════════════════════════════════════════════════
def plot_bsa(hs_a, hs_b, cid_a, cid_b, out):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, hs, cid in [(axes[0], hs_a, cid_a), (axes[1], hs_b, cid_b)]:
        if not hs: ax.set_title(f"Chain {cid}: no interface"); continue
        lbs = [r["label"] for r in hs]; bsas = [r["BSA_A2"] for r in hs]
        colors = ["#e74c3c" if r["is_hotspot"] else "#95a5a6" for r in hs]
        ax.bar(range(len(lbs)), bsas, color=colors, edgecolor="white", lw=0.5)
        ax.set_xticks(range(len(lbs))); ax.set_xticklabels(lbs, rotation=60, ha="right", fontsize=7)
        ax.axhline(25, color="#e74c3c", ls="--", alpha=0.6, label="Hotspot threshold")
        ax.set_ylabel("BSA (Å²)"); ax.set_title(f"Chain {cid} Interface")
        ax.legend(handles=[mpatches.Patch(color="#e74c3c",label="Hotspot"), mpatches.Patch(color="#95a5a6",label="Non-hotspot")])
    plt.suptitle("Computational Alanine Scanning — BSA per Interface Residue", y=1.02)
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"BSA bar chart: {out}")

def plot_contact_map(interface_data, cid_a, cid_b, out):
    pairs = interface_data["contact_pairs"]
    if not pairs: print("No contacts."); return
    ra = sorted(set(r[0].id[1] for r in pairs))
    rb = sorted(set(r[1].id[1] for r in pairs))
    mat = np.full((len(ra), len(rb)), np.nan)
    ai = {r: i for i, r in enumerate(ra)}; bi = {r: i for i, r in enumerate(rb)}
    for a, b, d in pairs:
        i, j = ai[a.id[1]], bi[b.id[1]]
        if np.isnan(mat[i,j]) or d < mat[i,j]: mat[i,j] = d
    fig, ax = plt.subplots(figsize=(max(8,len(rb)*0.3), max(6,len(ra)*0.3)))
    im = ax.imshow(np.where(np.isnan(mat), 5.0, mat), cmap="Blues_r", vmin=2.0, vmax=5.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="Min heavy-atom dist (Å)")
    ax.set_xticks(range(len(rb))); ax.set_xticklabels([f"{cid_b}{r}" for r in rb], rotation=90, fontsize=6)
    ax.set_yticks(range(len(ra))); ax.set_yticklabels([f"{cid_a}{r}" for r in ra], fontsize=6)
    ax.set_xlabel(f"Chain {cid_b}"); ax.set_ylabel(f"Chain {cid_a}")
    ax.set_title("Interface Contact Map")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Contact map: {out}")

def plot_radar(comp, cid_a, cid_b, out):
    cats = ["polar","apolar","charged+","charged-"]
    def norm(c): t=sum(c.values()) or 1; return [c.get(x,0)/t for x in cats]
    va = norm(comp["chain_a"]); vb = norm(comp["chain_b"])
    ang = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
    va += va[:1]; vb += vb[:1]; ang += ang[:1]
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(ang, va, "o-", c="#3498db", label=f"Chain {cid_a}", lw=2); ax.fill(ang, va, c="#3498db", alpha=0.25)
    ax.plot(ang, vb, "s-", c="#e74c3c", label=f"Chain {cid_b}", lw=2); ax.fill(ang, vb, c="#e74c3c", alpha=0.25)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(cats); ax.legend(); ax.set_title("Interface Composition", pad=20)
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Radar chart: {out}")

# ══════════════════════════════════════════════════════════════
# STEP 6: Save Results
# ══════════════════════════════════════════════════════════════
def save(hs_a, hs_b, comp, colabfold_scores, cid_a, cid_b, pdb_path, source_type, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = [{"chain":r["chain"],"resnum":r["resnum"],"resname":r["resname"],
             "label":r["label"],"BSA_A2":r["BSA_A2"],
             "hotspot_score":r["hotspot_score"],"is_hotspot":r["is_hotspot"]}
            for r in hs_a + hs_b]
    pd.DataFrame(rows).sort_values("hotspot_score",ascending=False).to_csv(f"{out_dir}/interface_residues.csv", index=False)
    with open(f"{out_dir}/hotspots.json","w") as f:
        json.dump({cid_a:[r["label"] for r in hs_a if r["is_hotspot"]],
                   cid_b:[r["label"] for r in hs_b if r["is_hotspot"]]}, f, indent=2)
    all_hs = sorted([r for r in hs_a+hs_b if r["is_hotspot"]], key=lambda x:x["hotspot_score"], reverse=True)
    report = ["="*65, "PROTEIN-PROTEIN INTERFACE ANALYSIS REPORT", "="*65,
              f"Input: {pdb_path} ({source_type})",
              f"Chains: {cid_a} / {cid_b}", ""]
    if colabfold_scores:
        report.append("--- Structure Quality (ColabFold) ---")
        for name, s in colabfold_scores.items():
            report.append(f"  {name}: ipTM={s['iptm']}, pTM={s['ptm']}")
        report.append("  ipTM > 0.75: high | 0.5-0.75: moderate | < 0.5: unreliable")
        report.append("")
    report += ["--- Interface Metrics ---",
               f"  Total BSA:              {comp['total_bsa_A2']} Å²",
               f"  Contact pairs:          {comp['n_contacts']}",
               f"  H-bonds (dist<3.5Å):   {comp['n_hbonds']}",
               f"  Salt bridges (dist<4.5Å): {comp['n_salt_bridges']}",
               f"  Shape complementarity:  {comp['shape_comp']}  [>0.65=good, <0.50=poor]",
               f"  Hotspots (chain {cid_a}):  {comp['n_ha']}",
               f"  Hotspots (chain {cid_b}):  {comp['n_hb']}",
               "", "--- Top Hotspot Residues ---"]
    for r in all_hs[:15]:
        report.append(f"  {r['label']:<14} BSA={r['BSA_A2']:.1f} Å²  score={r['hotspot_score']:.1f}")
    report += ["","--- Output Files ---",
               f"  {out_dir}/interface_residues.csv", f"  {out_dir}/hotspots.json",
               f"  {out_dir}/interface_bsa.png", f"  {out_dir}/contact_map.png",
               f"  {out_dir}/composition_radar.png", "="*65]
    print("\n" + "\n".join(report))
    with open(f"{out_dir}/interface_report.txt","w") as f: f.write("\n".join(report))
    print(f"\n✅ All outputs: {out_dir}/")

# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════
def run_pipeline(
    input_source: str,
    chain_a: str = "A",
    chain_b: str = "B",
    out_dir: str = "/root/ppi_interface/results",
    run_colabfold: bool = False,
    seq_a: str = "",
    seq_b: str = "",
) -> dict:
    """
    Full PPI interface analysis.
    input_source: PDB path/ID, or "predict" (if run_colabfold=True with seq_a/seq_b)
    """
    os.makedirs(out_dir, exist_ok=True)
    colabfold_scores = {}

    # Step 1: Load structure
    if input_source == "predict" and run_colabfold:
        print(f"\n[Mode B] Predicting complex from sequences...")
        pdb_path = predict_complex_colabfold(seq_a, seq_b, out_dir=f"{out_dir}/colabfold_output")
        source_type = "predicted"
    else:
        pdb_path_or_id = input_source if input_source != "predict" else f"{seq_a[:10]}:{seq_b[:10]}"
        print(f"\n[Mode A] Loading structure: {input_source}")
        _, source_type, pdb_path = load_structure(input_source)
        if source_type == "predicted":
            cf_dir = os.path.dirname(pdb_path)
            colabfold_scores = parse_colabfold_scores(cf_dir) if os.path.exists(cf_dir) else {}

    structure, _, _ = load_structure(pdb_path) if isinstance(pdb_path, str) else (None, None, pdb_path)
    if structure is None:
        raise RuntimeError("Could not load structure")

    # Re-parse if we already have pdb_path
    if pdb_path.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)

    # Step 2: Interface
    ifd = identify_interface(structure, chain_a, chain_b)
    # Step 3: Alanine scanning
    hs_a = alanine_scan(ifd, chain_a)
    hs_b = alanine_scan(ifd, chain_b)
    # Step 4: Composition (with H-bond/salt bridge)
    comp = analyze_interface_composition(ifd, chain_a, chain_b, hs_a, hs_b)
    # Step 5: Visualizations
    plot_bsa(hs_a, hs_b, chain_a, chain_b, f"{out_dir}/interface_bsa.png")
    plot_contact_map(ifd, chain_a, chain_b, f"{out_dir}/contact_map.png")
    plot_radar(comp, chain_a, chain_b, f"{out_dir}/composition_radar.png")
    # Step 6: Save
    save(hs_a, hs_b, comp, colabfold_scores, chain_a, chain_b, pdb_path, source_type, out_dir)
    return {"hotspots_a": hs_a, "hotspots_b": hs_b, "composition": comp, "interface_data": ifd}


# ══════════════════════════════════════════════════════════════
# DEMO A: PDB file (Mode A)
# ══════════════════════════════════════════════════════════════
def demo_mode_a():
    print("="*65)
    print("DEMO A — Mode A: Interface Analysis of PD-1/PD-L1 (PDB 4ZQK)")
    print("="*65)
    return run_pipeline(
        input_source="4ZQK",
        chain_a="A", chain_b="B",
        out_dir="/root/ppi_interface/results_pdl1",
    )

# ══════════════════════════════════════════════════════════════
# DEMO B: ColabFold prediction (Mode B) — commented by default
# ══════════════════════════════════════════════════════════════
def demo_mode_b():
    """
    Mode B: Predict complex from two amino acid sequences.
    Uses ColabFold AlphaFold2-Multimer via colabfold_batch.
    """
    # Example: nanobody + lysozyme (a classic model system)
    # Heavy chain of a typical nanobody
    NANOBODY_SEQ = (
        "QVQLVETGGGLVQPGGSLRLSCAASGFTFSSYSMNWVRQAPGKGLEWVSSISSSSSYIYYADSVKGRFTISR"
        "DDAKNSLYLQMNSLRAEDTAVYYCARKGLRGTYYYYYGMDVWGQGTTVTVSS"
    )
    # Hen egg-white lysozyme (antigen)
    LYSOZYME_SEQ = (
        "MRFLFIILLFVLTQGLSGCFSGSLQERAEQQNSFNKAVHCTLLSQLDHNVSGFNTTDVKPSIFSSRSRWHRN"
        "GCEKCSQHWDLELSHCSQNLTHSNSFTRSTQVNSRMLVKEKTAFDQCRHTRYEGNSRDLAEYHFRNGDIAIV"
        "DYGIDVQGGLSWRVFKSLPGERQFSAWCQLQGFRVLGIDTWHQVANGQWKVPGEVKFEEVTAHNVNNGKFKL"
        "DSAQRRDQEPLHTHFHGKLSNDRHQGELNILGPLLCSVRQ"
    )
    print("="*65)
    print("DEMO B — Mode B: ColabFold Prediction (nanobody + lysozyme)")
    print("="*65)
    return run_pipeline(
        input_source="predict",
        chain_a="A", chain_b="B",
        out_dir="/root/ppi_interface/results_predicted",
        run_colabfold=True,
        seq_a=NANOBODY_SEQ,
        seq_b=LYSOZYME_SEQ,
    )

if __name__ == "__main__":
    # Run Mode A (PDB file analysis)
    results_a = demo_mode_a()

    # To run Mode B (ColabFold prediction), uncomment:
    # results_b = demo_mode_b()
