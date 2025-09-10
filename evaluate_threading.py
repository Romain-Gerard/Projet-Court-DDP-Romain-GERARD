"""
evaluate_threading.py — Évaluation d'un threading par double programmation dynamique
(1 séquence cible + plusieurs gabarits listés dans manif.csv)

Entrée :
  - --manifest : CSV avec colonnes obligatoires
        pdb_path, chain, label, name, native
      où label = 1 (positif, même pli) / 0 (négatif)
         native = 1 optionnel pour marquer le 'vrai' gabarit
  - --fasta : séquence cible
  - --dope : dope.par
  - (option) --target-pdb / --target-chain : structure native de la cible (pour GDT-like)
  - hyperparamètres du pipeline (cycles, fenêtre, cutoff, biais/Beta SAP)
  - --jobs : nombre de processus en parallèle

Sorties :
  - <outdir>/per_template_results.csv (scores par gabarit)
  - <outdir>/summary.csv (Top-1/Top-5, AUC, rang du natif, Spearman)
  - <outdir>/roc.png, <outdir>/pr.png
"""

from __future__ import annotations
import argparse
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Matplotlib headless (évite blocages en environnement sans display)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Algorithme de superposition Kabsch sur matrices NumPy
from Bio.SVDSuperimposer import SVDSuperimposer

# ==== Import des briques depuis ton script principal ====
# Assure-toi que main.py est dans le PYTHONPATH (même dossier recommandé).
from main import (
    # I/O & prétraitements
    load_dope_par, read_fasta_one, encode_sequence,
    load_template_ca, build_distance_matrix,
    # Threading + scoring
    iterate_ddp, score_threading_energy, burial_proxy_counts,
    generate_decoy_energies, zscore_from_decoys,
    # Constantes (pour valeurs par défaut)
    K0_SELECTION, N_CYCLES, EXPAND_K,
    DEFAULT_Q_WINDOW, PAIRWISE_CUTOFF,
    ALPHA_BIAS, GAMMA_BIAS,
    SAP_BETA, SAP_NORM,
)

LOGGER = logging.getLogger("evaluate-threading")
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)

# ======================================================
#                 MÉTRIQUES UTILITAIRES
# ======================================================

def auc_trapezoid(x: np.ndarray | List[float], y: np.ndarray | List[float]) -> float:
    """Aire sous la courbe par intégration trapézoïdale (x trié croissant)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size == 0:
        return float("nan")
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))

def roc_curve(scores: np.ndarray | List[float], labels: np.ndarray | List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """ROC : FPR/TPR en balayant les seuils sur 'scores' (plus grand = plus positif)."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    order = np.argsort(-s)  # décroissant
    s, y = s[order], y[order]
    P = int(y.sum())
    N = int(len(y) - P)
    tpr = [0.0]; fpr = [0.0]
    tp = 0; fp = 0
    prev = math.inf
    for si, yi in zip(s, y):
        if si != prev:
            tpr.append(tp / P if P else 0.0)
            fpr.append(fp / N if N else 0.0)
            prev = si
        if yi == 1:
            tp += 1
        else:
            fp += 1
    tpr.append(tp / P if P else 0.0)
    fpr.append(fp / N if N else 0.0)
    return np.array(fpr), np.array(tpr)

def pr_curve(scores: np.ndarray | List[float], labels: np.ndarray | List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Precision-Recall (plus adapté quand les positifs sont rares)."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    order = np.argsort(-s)
    y = y[order]
    P = int(y.sum())
    tp = 0; fp = 0
    rec = []; prec = []
    for yi in y:
        if yi == 1: tp += 1
        else: fp += 1
        rec.append(tp / P if P else 0.0)
        prec.append(tp / (tp + fp))
    return np.array(rec), np.array(prec)

def spearman_rho(x: np.ndarray | List[float], y: np.ndarray | List[float]) -> float:
    """Corrélation de Spearman (rangs + Pearson). Ignore les NaN."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]; y = y[mask]
    # Rangs (moyenne en cas d'ex-aequo)
    def ranks(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a)
        r = np.empty_like(a, float)
        r[order] = np.arange(a.size, dtype=float)
        # moyenne des rangs pour valeurs identiques
        vals, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        accum = np.add.reduceat(r[order], np.r_[0, np.cumsum(counts[:-1])])
        meanr = accum / counts
        return meanr[inv]
    rx = ranks(x); ry = ranks(y)
    rx = (rx - rx.mean()) / (rx.std(ddof=1) + 1e-12)
    ry = (ry - ry.mean()) / (ry.std(ddof=1) + 1e-12)
    return float(np.mean(rx * ry))

# ======================================================
#           GDT-TS-LIKE (OPTIONNEL SI CIBLE NATIVE)
# ======================================================

def gdt_ts_like_from_alignment(aln_path: List[Tuple[int | None, int | None]],
                               coords_target: np.ndarray,
                               coords_template: np.ndarray) -> float:
    """
    Construit un 'modèle fileté' (Cα du gabarit aux positions alignées),
    superpose sur la cible native, calcule GDT_TS-like = mean(frac < {1,2,4,8} Å).
    """
    pairs = [(i, j) for (i, j) in aln_path if i is not None and j is not None]
    if len(pairs) < 5:
        return float("nan")
    i_tmpl = np.asarray([p[0] for p in pairs], int)
    j_tgt  = np.asarray([p[1] for p in pairs], int)
    P = np.asarray(coords_template[i_tmpl], float)  # moving
    Q = np.asarray(coords_target[j_tgt],   float)  # fixed
    sup = SVDSuperimposer(); sup.set(P, Q); sup.run()
    rot, tran = sup.get_rotran()
    P_aln = P.dot(rot) + tran
    d = np.linalg.norm(P_aln - Q, axis=1)
    thr = (1.0, 2.0, 4.0, 8.0)
    return float(np.mean([(d < t).mean() for t in thr]))

# ======================================================
#                    WORKER PARALLÈLE
# ======================================================

def _worker_one_template(entry: Dict[str, Any],
                         dope: np.ndarray,
                         seq_idx: np.ndarray,
                         coords_target: np.ndarray | None,
                         pipe_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Évalue 1 gabarit (process isolé) : threading itératif, énergie, z, (option) GDT-like.
    """
    # Couper le bruit des logs côté worker
    try:
        logging.getLogger("ddp-threading-s1s2").setLevel(logging.ERROR)
    except Exception:
        pass

    t0 = time.time()

    pdb_path = Path(entry["pdb_path"])
    chain    = entry.get("chain") or None
    label    = int(entry.get("label", 0))
    name     = str(entry.get("name", pdb_path.stem))
    is_native= int(entry.get("native", 0))

    # Charger le gabarit et distances
    coords_tmpl, _res_ids = load_template_ca(pdb_path, chain_id=chain)
    D = build_distance_matrix(coords_tmpl)

    # Threading itératif (avec/sans SAP)
    aln_path, score_global, _H_final = iterate_ddp(
        dope, D, seq_idx, sse=None,
        K0=pipe_params["K0"], cycles=pipe_params["cycles"], expand_k=pipe_params["expand_k"],
        q_window=pipe_params["q_window"], cutoff=pipe_params["cutoff"],
        alpha_bias=pipe_params["alpha_bias"], gamma_bias=pipe_params["gamma_bias"],
        use_sap=pipe_params["use_sap"], beta_sap=pipe_params["sap_beta"], sap_norm=pipe_params["sap_norm"],
    )

    # Énergie + z
    bur = burial_proxy_counts(D)
    E = score_threading_energy(aln_path, dope, D, seq_idx,
                               cutoff=pipe_params["cutoff"],
                               burial_counts=bur, w_burial=0.0)
    decoys = generate_decoy_energies(aln_path, dope, D, seq_idx,
                                     n_decoys=pipe_params["n_decoys"],
                                     cutoff=pipe_params["cutoff"])
    z = zscore_from_decoys(E, decoys)

    # (option) GDT-like
    if coords_target is not None:
        gdt_like = gdt_ts_like_from_alignment(aln_path, coords_target, coords_tmpl)
    else:
        gdt_like = float("nan")

    return {
        "name": name,
        "pdb_path": str(pdb_path),
        "chain": chain if chain is not None else "",
        "label": label,
        "native": is_native,
        "score_global": float(score_global),
        "E_final": float(E),
        "z": float(z),
        "gdt_like": float(gdt_like),
        "aligned_len": int(sum(1 for t in aln_path if t[0] is not None and t[1] is not None)),
        "runtime_sec": float(time.time() - t0),
    }

# ======================================================
#                  BOUCLE D'ÉVALUATION
# ======================================================

def run_eval(manifest_csv: Path,
             fasta_path: Path,
             dope_path: Path,
             outdir: Path,
             use_sap: bool = False,
             sap_beta: float = SAP_BETA,
             sap_norm: str = SAP_NORM,
             cycles: int = N_CYCLES,
             expand_k: int = EXPAND_K,
             q_window: int = DEFAULT_Q_WINDOW,
             cutoff: float = PAIRWISE_CUTOFF,
             alpha_bias: float = ALPHA_BIAS,
             gamma_bias: float = GAMMA_BIAS,
             target_pdb: Path | None = None,
             target_chain: str | None = None,
             n_decoys: int = 50,
             jobs: int = 1,) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    outdir.mkdir(parents=True, exist_ok=True)

    # Charger DOPE + séquence cible
    LOGGER.info("Chargement DOPE depuis %s", dope_path)
    dope = load_dope_par(dope_path)
    LOGGER.info("Lecture FASTA cible : %s", fasta_path)
    seq_str = read_fasta_one(fasta_path)
    seq_idx = encode_sequence(seq_str)

    # (Option) coords natifs de la cible pour GDT-like
    coords_target = None
    if target_pdb is not None:
        LOGGER.info("Chargement cible native : %s (chain=%s)", target_pdb, target_chain or "auto")
        coords_target, _ = load_template_ca(target_pdb, chain_id=target_chain)

    # Lire le manifeste
    df = pd.read_csv(manifest_csv)
    required_cols = {"pdb_path", "chain", "label", "name", "native"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"manif.csv doit contenir : {sorted(required_cols)}")

    # Préparer les entrées pour les workers
    entries = []
    for _, row in df.iterrows():
        entries.append({
            "pdb_path": str(row["pdb_path"]),
            "chain": None if pd.isna(row.get("chain", None)) else str(row["chain"]),
            "label": int(row["label"]),
            "name": str(row.get("name", Path(str(row["pdb_path"])).stem)),
            "native": int(row.get("native", 0)),
        })

    pipe_params = dict(
        use_sap=use_sap, sap_beta=sap_beta, sap_norm=sap_norm,
        cycles=cycles, expand_k=expand_k,
        q_window=q_window, cutoff=cutoff,
        alpha_bias=alpha_bias, gamma_bias=gamma_bias,
        n_decoys=n_decoys, K0=K0_SELECTION,
    )

    # Éviter la sur-subscription BLAS quand jobs>1
    if jobs and jobs > 1:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Exécution (parallèle si jobs>1)
    LOGGER.info("Évaluation de %d gabarits (%d jobs).", len(entries), jobs)
    rows: List[Dict[str, Any]] = []
    if jobs and jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(_worker_one_template, e, dope, seq_idx, coords_target, pipe_params)
                    for e in entries]
            for fut in as_completed(futs):
                rows.append(fut.result())
    else:
        for e in entries:
            rows.append(_worker_one_template(e, dope, seq_idx, coords_target, pipe_params))

    res = pd.DataFrame(rows)
    res.to_csv(outdir / "per_template_results.csv", index=False)
    LOGGER.info("Résultats par gabarit écrits dans %s", outdir / "per_template_results.csv")

    # ---- Agrégats / métriques de classement ----
    # Score de décision: plus grand => plus positif. z étant "plus négatif = meilleur", on prend -z.
    res["decision_score"] = -res["z"]

    # Top-1 / Top-5
    res_sorted = res.sort_values("decision_score", ascending=False)
    top1_is_pos = int(res_sorted.iloc[0]["label"] == 1) if len(res_sorted) else 0
    top5_is_pos = int((res_sorted.iloc[:5]["label"] == 1).any()) if len(res_sorted) else 0

    # Rang du natif
    rank_native = float("nan")
    if (res.get("native", pd.Series(dtype=int)) == 1).any():
        pos = np.where(res_sorted["native"].to_numpy(dtype=int) == 1)[0]
        rank_native = int(pos[0] + 1) if len(pos) else float("nan")

    # ROC / PR
    fpr, tpr = roc_curve(res["decision_score"].to_numpy(), res["label"].to_numpy())
    roc_auc = auc_trapezoid(fpr, tpr)
    rec, prec = pr_curve(res["decision_score"].to_numpy(), res["label"].to_numpy())
    pr_auc = auc_trapezoid(rec, prec)

    # Spearman (si GDT-like dispo)
    rho = spearman_rho(res["decision_score"].to_numpy(), res["gdt_like"].to_numpy())

    summary = {
        "n_templates": int(len(res)),
        "n_pos": int(res["label"].sum()),
        "n_neg": int((1 - res["label"]).sum()),
        "top1_accuracy": int(top1_is_pos),
        "top5_accuracy": int(top5_is_pos),
        "rank_native": rank_native,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "spearman_score_vs_gdt": float(rho),
    }
    pd.DataFrame([summary]).to_csv(outdir / "summary.csv", index=False)
    LOGGER.info("Résumé écrit dans %s", outdir / "summary.csv")

    # Courbes ROC / PR
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={roc_auc:.3f})")
    plt.tight_layout(); plt.savefig(outdir / "roc.png", dpi=150); plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (AUC={pr_auc:.3f})")
    plt.tight_layout(); plt.savefig(outdir / "pr.png", dpi=150); plt.close()

    return res, summary

# ======================================================
#                           CLI
# ======================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Évaluation threading (1 cible + multiples gabarits).")
    ap.add_argument("--manifest", required=True, type=Path, help="CSV: pdb_path,chain,label,name,native")
    ap.add_argument("--fasta",   required=True, type=Path, help="Séquence cible (FASTA)")
    ap.add_argument("--dope",    required=True, type=Path, help="Fichier dope.par")
    ap.add_argument("--outdir",  default=Path("./eval_out"), type=Path)

    # Pipeline
    ap.add_argument("--sap-reinforce", action="store_true", help="Activer renforcement SAP-like")
    ap.add_argument("--sap-beta", type=float, default=SAP_BETA)
    ap.add_argument("--sap-norm", type=str, default=SAP_NORM, choices=["length", "none"])
    ap.add_argument("--cycles",   type=int, default=N_CYCLES)
    ap.add_argument("--expand-k", type=int, default=EXPAND_K)
    ap.add_argument("--q-window", type=int, default=DEFAULT_Q_WINDOW)
    ap.add_argument("--cutoff",   type=float, default=PAIRWISE_CUTOFF)
    ap.add_argument("--alpha-bias", type=float, default=ALPHA_BIAS)
    ap.add_argument("--gamma-bias", type=float, default=GAMMA_BIAS)
    ap.add_argument("--n-decoys", type=int, default=50)

    # (Option) structure native de la cible pour GDT-like
    ap.add_argument("--target-pdb", type=Path, default=None)
    ap.add_argument("--target-chain", type=str, default=None)

    # Parallélisation
    ap.add_argument("--jobs", "-j", type=int, default=1, help="Nb de processus en parallèle (1=mono)")

    return ap

def main():
    parser = build_parser()
    args = parser.parse_args()

    run_eval(
        manifest_csv=args.manifest,
        fasta_path=args.fasta,
        dope_path=args.dope,
        outdir=args.outdir,
        use_sap=args.sap_reinforce,
        sap_beta=args.sap_beta,
        sap_norm=args.sap_norm,
        cycles=args.cycles,
        expand_k=args.expand_k,
        q_window=args.q_window,
        cutoff=args.cutoff,
        alpha_bias=args.alpha_bias,
        gamma_bias=args.gamma_bias,
        target_pdb=args.target_pdb,
        target_chain=args.target_chain,
        n_decoys=args.n_decoys,
        jobs=args.jobs,
    )

if __name__ == "__main__":
    main()
