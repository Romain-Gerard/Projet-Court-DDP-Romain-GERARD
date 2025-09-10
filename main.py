"""DDP Threading."""



import logging
from pathlib import Path
import argparse
import sys, shutil, time

import numpy as np
from Bio import SeqIO, PDB
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



# =======================
# Globals / Constants
# =======================

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOGGER = logging.getLogger("ddp-threading-s1s2")

# --- I/O & sorties par défaut ---
DEFAULT_OUTDIR = Path("./ddp_out")
PLOT_FIGSIZE = (8, 6)

# --- Paramètres DOPE / discrétisation des distances ---
BIN_WIDTH = 0.5          # largeur du bin (Å)
N_BINS = 30              # nb de bins (0..29) → 0..15 Å
PAIRWISE_CUTOFF = 10.0   # cutoff des paires Cα-Cα pour l'énergie finale (Å)

# --- Tables acides aminés (1 lettre ↔ 3 lettres ↔ index 0..19) ---
_AA_TABLE = [
    ("A", "ALA"), ("R", "ARG"), ("N", "ASN"), ("D", "ASP"), ("C", "CYS"),
    ("Q", "GLN"), ("E", "GLU"), ("G", "GLY"), ("H", "HIS"), ("I", "ILE"),
    ("L", "LEU"), ("K", "LYS"), ("M", "MET"), ("F", "PHE"), ("P", "PRO"),
    ("S", "SER"), ("T", "THR"), ("W", "TRP"), ("Y", "TYR"), ("V", "VAL"),
]
AA1_TO_INDEX  = {a1: i for i, (a1, a3) in enumerate(_AA_TABLE)}
AA3_TO_INDEX  = {a3: i for i, (a1, a3) in enumerate(_AA_TABLE)}
INDEX_TO_AA1  = {i: a1 for i, (a1, a3) in enumerate(_AA_TABLE)}
VALID_AA1     = set(AA1_TO_INDEX.keys())

# =======================
# DP local (Smith-Waterman) — Sprints 5-7
# =======================
DEFAULT_Q_WINDOW = 12   # demi-fenêtre en séquence pour L_{m,n}
GAP_OPEN_LOCAL   = 6.0  # pénalité d'ouverture (unités de score local)
GAP_EXT_LOCAL    = 0.5  # pénalité d'extension

# =======================
# DP global (Needleman-Wunsch/Gotoh) — Sprints 8-9
# =======================
K0_SELECTION   = 30     # # de paires (m,n) initiales pour construire Q
# Gaps horizontaux (dans la SEQUENCE)
GAP_OPEN_H     = 7.0
GAP_EXT_H      = 0.6
# Gaps verticaux (dans la STRUCTURE) — SSE ≠ boucle
GAP_OPEN_V_LOOP = 6.0
GAP_EXT_V_LOOP  = 0.5
GAP_OPEN_V_SSE  = 12.0  # hélices/brins plus coûteux
GAP_EXT_V_SSE   = 2.0

# =======================
# Orchestration itérative — Sprint 10
# =======================
N_CYCLES       = 50     # nb max de cycles
EXPAND_K       = 40     # # de nouveaux pics H ajoutés par cycle
ALN_NEIGH_WIN  = 2      # demi-fenêtre autour du chemin pour élargir la sélection
ALPHA_BIAS     = 0.3    # H = Q + ALPHA_BIAS * Bias + β * R
GAMMA_BIAS     = 0.7    # Bias <- GAMMA_BIAS * Bias + (1-γ) * norm(H)
MIN_IMPROV     = 1e-4   # arrêt si amélioration < MIN_IMPROV (et couverture OK)

# =======================
# Renforcement SAP-like (option) — utilisé dans H si activé
# =======================
SAP_REINFORCE_DEFAULT = False
SAP_BETA  = 0.5         # poids β devant R
SAP_NORM  = "length"    # "length" (divise par longueur du chemin local) ou "none"

# =======================
# Scoring final & statistique — Sprint 11
# =======================
BURIAL_RADIUS = 8.0   # rayon (Å) pour proxy d'enterrement
W_BURIAL      = 0.0   # poids du terme d'enterrement (0 = désactivé)
N_DECOYS      = 200   # nb de décoys pour z-score

# (Option) Hydropathie Kyte-Doolittle si W_BURIAL > 0
_KD = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9,
    "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3,
    "P": -1.6, "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5,
    "K": -3.9, "R": -4.5
}
KD_VEC = np.array([_KD[a1] for a1, _ in _AA_TABLE], dtype=np.float64)



# =======================
# 1 — DOPE loader
# =======================

def load_dope_par(path: Path) -> np.ndarray:
    """
    Lire un fichier **Modeller DOPE** (`dope.par`) et construire un tenseur
    de potentiels **Cα-Cα** indexé par (AA_i, AA_j, bin de distance).

    Parameters
    ----------
    path : pathlib.Path or str
        Chemin vers le fichier `dope.par`.

    Returns
    -------
    dope : numpy.ndarray, shape (20, 20, N_BINS), dtype float64
        Tenseur tel que `dope[i, j, k]` est le **potentiel** (énergie statistique)
        pour la paire d'acides aminés (AA_i, AA_j) au **bin de distance** `k`.
        - Index AA : 0..19 (voir `AA3_TO_INDEX`, `AA1_TO_INDEX`)
        - Bins : 0..N_BINS-1, avec largeur `BIN_WIDTH` (Å).
          Le bin `k` approxime l'intervalle `[k*BIN_WIDTH, (k+1)*BIN_WIDTH)`.

    Raises
    ------
    FileNotFoundError
        Si `path` n'existe pas.

    Notes
    -----
    - **Seules** les lignes `CA`-`CA` sont conservées (atomes Cα).
    - On **symétrise** le tenseur : si `(i,j)` existe mais pas `(j,i)`,
      on copie `dope[i,j,:]` vers `dope[j,i,:]`.
    - Les valeurs DOPE : plus **négatif** ≈ plus **favorable** (score).
    - Les paires totalement **absentes** restent à 0. Un warning est émis.
    - Lecture robuste : lignes vides/commentées ignorées ; valeurs non numériques
      ou lignes incomplètes **skippées** proprement.

    Examples
    --------
    >>> dope = load_dope_par(Path("dope.par"))
    >>> dope.shape
    (20, 20, 30)
    >>> dope[AA1_TO_INDEX["A"], AA1_TO_INDEX["A"], :5]
    array([...])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dope.par introuvable: {path}")

    # Tenseur de sortie + masque des paires effectivement vues dans le fichier
    dope = np.zeros((20, 20, N_BINS), dtype=np.float64)
    seen = np.zeros((20, 20), dtype=bool)

    # Lecture ligne à ligne (flux) — évite de charger tout en mémoire
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line_no, line in enumerate(fh, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue  # vide ou commentaire

            parts = s.split()
            # Format attendu minimal : RES1 ATOM1 RES2 ATOM2 + N_BINS valeurs
            if len(parts) < 4 + N_BINS:
                # Ligne incomplète (fichiers coupés / colonnes tronquées)
                continue

            res1_raw, atom1_raw, res2_raw, atom2_raw = parts[0], parts[1], parts[2], parts[3]
            atom1 = atom1_raw.upper()
            atom2 = atom2_raw.upper()
            if atom1 != "CA" or atom2 != "CA":
                # On ne retient QUE les potentiels Cα-Cα
                continue

            res1 = res1_raw.upper()
            res2 = res2_raw.upper()
            if res1 not in AA3_TO_INDEX or res2 not in AA3_TO_INDEX:
                # Résidus non standards / inconnus
                continue

            # Conversion numérique des N_BINS colonnes de potentiel
            try:
                values = [float(x) for x in parts[4:4 + N_BINS]]
            except ValueError:
                # Valeur non numérique
                continue
            if len(values) != N_BINS:
                # Sécurité
                continue

            i = AA3_TO_INDEX[res1]
            j = AA3_TO_INDEX[res2]
            dope[i, j, :] = np.asarray(values, dtype=np.float64)
            seen[i, j] = True

    # Symétrisation : si (i,j) vu mais pas (j,i), on copie
    for i in range(20):
        for j in range(20):
            if seen[i, j] and not seen[j, i]:
                dope[j, i, :] = dope[i, j, :]
                seen[j, i] = True

    # Vérification : si certaines paires manquent totalement
    if not seen.all():
        missing = np.argwhere(~seen)
        # On log seulement les premières pour ne pas spammer
        LOGGER.warning("Certaines paires AA Cα-Cα manquent dans dope.par (rare). "
                       "Exemples d'indices manquants (i,j): %s", missing[:5])

    return dope



# ===========================
# 2 — FASTA & encodage
# ===========================

def read_fasta_one(path: Path) -> str:
    """
    Lire un fichier FASTA et retourner la **première** séquence en acides aminés (1 lettre).

    Parameters
    ----------
    path : pathlib.Path or str
        Chemin vers le fichier FASTA.

    Returns
    -------
    str
        Séquence d'acides aminés en **1 lettre**, uppercase, sans symboles '*'.

    Raises
    ------
    FileNotFoundError
        Si le fichier FASTA n'existe pas.
    ValueError
        Si aucune séquence n'est trouvée ou si la première séquence est vide.

    Notes
    -----
    - Seule la **première** entrée du FASTA est lue (comportement intentionnel).
    - Les symboles de stop '*' sont retirés (`replace("*", "")`), pratique pour
      des FASTA génomiques traduits.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA introuvable: {path}")

    # On prend la première séquence et on la nettoie minimalement
    for rec in SeqIO.parse(str(path), "fasta"):
        seq = str(rec.seq).upper().replace("*", "")
        if not seq:
            raise ValueError(f"Séquence vide dans {path}")
        return seq

    # Si on arrive ici, aucun enregistrement n'a été lu
    raise ValueError(f"Aucune séquence lue dans {path}")


def encode_sequence(seq_1letter: str) -> np.ndarray:
    """
    Convertir une séquence 1-lettre en indices entiers [0..19] selon `AA1_TO_INDEX`.

    Parameters
    ----------
    seq_1letter : str
        Séquence d'acides aminés (1 lettre), insensible à la casse.

    Returns
    -------
    numpy.ndarray
        Tableau `int64` de longueur N, où chaque entrée est l'index 0..19 de l'AA.

    Raises
    ------
    ValueError
        Si la séquence contient des caractères non supportés.
        Les acides aminés valides sont : sorted(VALID_AA1).

    Notes
    -----
    - Conçu pour des séquences **protéiques standards** (20 AA).
    - Les caractères ambigus (ex. 'X', 'B', 'Z') **ne sont pas** acceptés : on préfère
      échouer tôt plutôt que propager de l'incertitude dans le threading.
    """
    # Normalisation simple (uppercase)
    seq_1letter = seq_1letter.upper()

    # Validation stricte : uniquement les 20 AA standards
    bad = sorted({a for a in set(seq_1letter) if a not in VALID_AA1})
    if bad:
        valid = "".join(sorted(VALID_AA1))
        raise ValueError(
            f"Acides aminés non supportés dans la séquence: {bad}. "
            f"Utiliser uniquement ces caractères: {valid}"
        )

    # Encodage vectorisé vers indices 0..19
    idx = np.fromiter((AA1_TO_INDEX[a] for a in seq_1letter),
                      dtype=np.int64, count=len(seq_1letter))
    return idx



# =======================
# 3 — PDB → Cα
# =======================

def load_template_ca(pdb_path: Path, chain_id: str | None = None) -> tuple[np.ndarray, list[tuple[str, int, str]]]:
    """
    Charger un gabarit PDB et extraire les coordonnées des **Cα** pour une chaîne.

    Parameters
    ----------
    pdb_path : pathlib.Path or str
        Chemin vers un fichier PDB (format PDB classique).  
        *(Si vous souhaitez supporter mmCIF, utilisez Bio.PDB.MMCIFParser à la place.)*
    chain_id : str or None, optional
        Identifiant de la chaîne à extraire. Si `None` et qu'il n'y a **qu'une seule**
        chaîne dans le premier modèle, elle est sélectionnée automatiquement.

    Returns
    -------
    coords : numpy.ndarray, shape (M, 3), dtype float64
        Coordonnées cartésiennes des atomes **Cα** dans l'ordre de la chaîne.
    res_ids : list[tuple[str, int, str]]
        Métadonnées parallèles de longueur M pour chaque Cα :  
        `(resname3, resseq, icode)` avec `resname3` en 3 lettres (ex. "LYS"),
        `resseq` le numéro de résidu PDB, `icode` le *insertion code* (chaîne
        vide si absent).

    Raises
    ------
    FileNotFoundError
        Si `pdb_path` n'existe pas.
    ValueError
        - Si le fichier ne contient aucun modèle.
        - Si plusieurs chaînes sont présentes et `chain_id` est `None`.
        - Si `chain_id` n'existe pas.
        - Si aucun Cα standard n'est trouvé pour la chaîne.

    Notes
    -----
    - Seul le **premier modèle** du fichier PDB est utilisé.
    - **AltLoc** : s'il existe des positions alternatives pour Cα, on choisit
      l'atome dont l'**occupancy** est la plus élevée.
    - Les résidus **non standards** (hétéro-atomes, ligands, acides aminés non
      canoniques) ou **sans Cα** sont ignorés.
    - Cette fonction ne renvoie **que** les Cα

    Examples
    --------
    >>> coords, res_ids = load_template_ca(Path("templates/1UBQ.pdb"), chain_id="A")
    >>> coords.shape
    (76, 3)
    >>> res_ids[0]
    ('MET', 1, '')
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"Fichier PDB introuvable : {pdb_path}")

    # Parseur PDB (on reste sur le format PDB classique ici)
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("tmpl", str(pdb_path))

    # Premier modèle
    try:
        model = next(structure.get_models())
    except StopIteration:
        raise ValueError(f"Aucun modèle trouvé dans {pdb_path}")

    # Choix de la chaîne
    chains = [c.id for c in model.get_chains()]
    chains_uniq = list(dict.fromkeys(chains))  # préserve l'ordre, dédoublonne
    if chain_id is None:
        if len(chains_uniq) == 1:
            chain_id = chains_uniq[0]
        else:
            raise ValueError(
                f"Plusieurs chaînes détectées {chains_uniq}. "
                f"Spécifiez --chain <ID>."
            )
    if chain_id not in chains_uniq:
        raise ValueError(f"Chaîne '{chain_id}' introuvable. Disponibles : {chains_uniq}")

    chain = model[chain_id]

    coords: list[np.ndarray] = []
    res_ids: list[tuple[str, int, str]] = []

    # Parcours séquentiel des résidus de la chaîne
    for res in chain.get_residues():
        # Ne garder que les acides aminés standards (Bio.PDB gère la plupart des cas)
        if not PDB.is_aa(res, standard=True):
            continue

        if "CA" not in res:
            continue

        atom = res["CA"]
        # Gestion des positions alternatives : choisir l'occupancy max
        if atom.is_disordered():
            alts = list(atom.child_dict.values())
            atom = max(alts, key=lambda a: (a.get_occupancy() or 0.0))

        # Coordonnées Cα
        xyz = atom.get_coord().astype(np.float64)
        coords.append(xyz)

        # Identité du résidu (format PDB : (het, resseq, icode))
        resname3 = res.get_resname().upper()
        resseq = int(res.id[1])
        icode = res.id[2] if isinstance(res.id[2], str) else ""
        res_ids.append((resname3, resseq, icode.strip()))

    if not coords:
        raise ValueError(f"Aucun Cα trouvé dans {pdb_path} chaîne {chain_id}")

    return np.asarray(coords, dtype=np.float64), res_ids



# ==============================
# 4 — Matrice de distance
# ==============================

def build_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Construire la matrice des distances **Cα-Cα** à partir des coordonnées.

    Parameters
    ----------
    coords : numpy.ndarray, shape (M, 3), dtype float64
        Coordonnées cartésiennes des Cα (ordre séquentiel).

    Returns
    -------
    D : numpy.ndarray, shape (M, M), dtype float64
        Matrice des distances euclidiennes Cα-Cα, **symétrique** avec
        `diag(D) = 0`.

    Raises
    ------
    ValueError
        Si `coords` n'a pas la forme `(M, 3)`.
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords doit être [M,3], reçu {coords.shape}")

    # Différences vectorisées
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=-1), dtype=np.float64)

    # Nettoyage numér.
    np.fill_diagonal(D, 0.0)
    return D



# =============================
# 5 — Binning & lookup DOPE
# =============================

def distance_to_bin(d: float, bin_width: float = BIN_WIDTH, n_bins: int = N_BINS) -> int:
    """
    Mapper une distance en Å vers l'indice de bin DOPE (0..n_bins-1).

    Parameters
    ----------
    d : float
        Distance Cα-Cα (Å), attendue ≥ 0.
    bin_width : float, optional
        Largeur d'un bin (Å).
    n_bins : int, optional
        Nombre total de bins.

    Returns
    -------
    int
        Index de bin `k` dans `[0, n_bins-1]`.

    Raises
    ------
    ValueError
        Si `d` est négative.
    """
    if d < 0:
        raise ValueError("distance négative")
    k = int(d / bin_width)
    if k >= n_bins:
        k = n_bins - 1
    return k


def dope_ca_pair(dope: np.ndarray, aa_i: int, aa_j: int, d: float,
                 cutoff: float = PAIRWISE_CUTOFF, as_similarity: bool = True) -> float:
    """
    Retourner la contribution DOPE pour une paire (aa_i, aa_j) à distance `d`.

    Parameters
    ----------
    dope : numpy.ndarray
        Tenseur DOPE de forme (20, 20, N_BINS). Valeurs = énergies (plus bas = meilleur).
    aa_i, aa_j : int
        Indices d'acides aminés (0..19) pour la séquence aux positions i et j.
    d : float
        Distance Cα-Cα (Å) pour la **paire structurale** considérée.
    cutoff : float, optional
        Au-delà de cette distance, la contribution est nulle (par défaut ~10 Å).
    as_similarity : bool, optional
        Si `True`, renvoie un **score à maximiser** (= `-énergie DOPE`).
        Si `False`, renvoie l'**énergie DOPE** telle quelle.

    Returns
    -------
    float
        Contribution DOPE (score si `as_similarity=True`, sinon énergie).

    Notes
    -----
    - Hypothèse : paires **Cα-Cα** uniquement (cohérent avec le tenseur chargé).
    - Si `d > cutoff`, retourne 0.0 (contribution nulle).
    """
    if d > cutoff:
        return 0.0
    k = distance_to_bin(d)
    e = float(dope[aa_i, aa_j, k])
    return -e if as_similarity else e  # on maximise le "score" par défaut



# ==============================================
# 6 — Voisinages & matrice d'environnement L
# ==============================================

def neighbors_within(D: np.ndarray, m: int, r: float = PAIRWISE_CUTOFF) -> list[int]:
    """
    Lister les voisins structuraux du résidu gabarit `m` dans un rayon `r`.

    Parameters
    ----------
    D : numpy.ndarray, shape (M, M)
        Matrice des distances Cα-Cα du gabarit.
    m : int
        Index du résidu gabarit (0..M-1).
    r : float, optional
        Rayon de voisinage (Å). Les p tels que D[m, p] ≤ r sont retenus.

    Returns
    -------
    list of int
        Indices `p` (p ≠ m) des voisins de `m` dans le gabarit.

    Notes
    -----
    Biologiquement : ce sont les résidus **structuralement proches** de `m`
    (en 3D), i.e. son environnement local. Ils serviront d'“ancrages” pour
    scorer l'hypothèse `(m ↔ n)` dans la matrice L.
    """
    dist_row = D[m]
    mask = (dist_row <= r)
    mask[m] = False  # on exclut m lui-même
    return np.nonzero(mask)[0].tolist()


def lowlevel_env_matrix(dope: np.ndarray,
                        D_tmpl: np.ndarray,
                        seq_idx: np.ndarray,
                        m: int,
                        n: int,
                        q_window: int | None = DEFAULT_Q_WINDOW,
                        cutoff: float = PAIRWISE_CUTOFF) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Construire la matrice d'environnement **L** pour l'hypothèse `(m ↔ n)`.

    L[ip, jq] = score (à maximiser) pour faire correspondre :
      - un **voisin structural** `p` de `m` (dans le gabarit), et
      - une **position** `q` dans la séquence,
    en utilisant le potentiel DOPE évalué à la distance d(m,p).

    Parameters
    ----------
    dope : numpy.ndarray, shape (20, 20, N_BINS)
        Tenseur DOPE Cα-Cα (énergies). On utilise `-énergie` comme **score**.
    D_tmpl : numpy.ndarray, shape (M, M)
        Matrice de distances Cα-Cα du gabarit (template).
    seq_idx : numpy.ndarray, shape (N,)
        Séquence cible encodée en indices 0..19 (voir `encode_sequence`).
    m : int
        Index (0..M-1) du résidu gabarit testé.
    n : int
        Index (0..N-1) de la position de séquence hypothétique.
    q_window : int or None, optional
        Demi-fenêtre autour de `n` pour restreindre les `q`. Si `None`, on
        considère **tous** les `q` (0..N-1).
    cutoff : float, optional
        Cutoff de distance (Å) pour la contribution DOPE (par défaut ~10 Å).

    Returns
    -------
    L : numpy.ndarray, shape (|P|, |Q|), dtype float64
        Matrice des **scores locaux** à maximiser (sera donnée au SW local).
    P_list : list of int
        Liste des **indices gabarit** `p` (voisins de `m`) correspondant aux lignes de L.
    Q_list : list of int
        Liste des **indices séquence** `q` correspondant aux colonnes de L.

    Notes
    -----
    - Biologie : on évalue la **compatibilité** de l'environnement 3D de `m`
      avec des fragments de la séquence autour de `n`.
    - Informatique : `L` est ensuite optimisée par **Smith-Waterman local**
      pour obtenir un meilleur score et un petit chemin `(p,q)` cohérent.
    - Complexité : O(|P|·|Q|) appels à DOPE (ici vectorisé au niveau “ligne”).
    """
    M = D_tmpl.shape[0]
    N = int(seq_idx.shape[0])

    # Voisins structuraux de m (dans le gabarit)
    P_list = neighbors_within(D_tmpl, m, r=cutoff)  # indices p dans [0..M-1]

    # Fenêtre de séquence autour de n (ou tous)
    if q_window is None:
        Q_list = list(range(N))
    else:
        q0 = max(0, n - q_window)
        q1 = min(N, n + q_window + 1)
        Q_list = list(range(q0, q1))

    if len(P_list) == 0 or len(Q_list) == 0:
        # Aucun voisin ou fenêtre vide → matrice 0x0 (score local sera 0)
        return np.zeros((0, 0), dtype=np.float64), P_list, Q_list

    # Prépare L : pour chaque p ∈ P_list et q ∈ Q_list
    L = np.empty((len(P_list), len(Q_list)), dtype=np.float64)

    # Pour un p fixé, d(m,p) est constant ; seul aa_j = seq[q] varie
    aa_i = int(seq_idx[n])  # AA de la séquence à la position n (hypothèse m↔n)
    q_idx = np.array(Q_list, dtype=int)
    aa_js = seq_idx[q_idx]

    for ip, p in enumerate(P_list):
        d_mp = float(D_tmpl[m, p])
        # Lookup DOPE pour tous q de Q_list (on renvoie un score à maximiser)
        row_vals = [dope_ca_pair(dope, aa_i, int(aa_j), d_mp, cutoff=cutoff, as_similarity=True)
                    for aa_j in aa_js]
        L[ip, :] = row_vals

    return L, P_list, Q_list



# ==============================================
# 7 — Smith-Waterman local (affine) → Q(m,n)
# ==============================================

def smith_waterman_local(L: np.ndarray,
                         gap_open: float = GAP_OPEN_LOCAL,
                         gap_ext: float  = GAP_EXT_LOCAL) -> tuple[float, list[tuple[int,int]]]:
    """
    Smith-Waterman **local** avec pénalités **affines** (Gotoh) sur une matrice de scores `L`.

    Parameters
    ----------
    L : numpy.ndarray, shape (P, Q)
        Matrice de **scores à maximiser** (issus de l'environnement local).
        L[ip, jq] évalue le match entre le voisin `p` et la position de séquence `q`.
    gap_open : float, optional
        Coût d'ouverture d'un gap (≥ 0).
    gap_ext : float, optional
        Coût d'extension d'un gap (≥ 0).

    Returns
    -------
    best_score : float
        Meilleur score local atteint dans `L` (≥ 0).
    path : list[tuple[int, int]]
        Chemin local optimal dans l'espace `(ip, jq)`, **indices dans `L`**.
        Liste vide si `best_score == 0`.

    Notes
    -----
    - Trois états de DP (Gotoh) :
        * `M` : match (on consomme `L[i-1, j-1]`)
        * `X` : gap **vertical** (avance en `i`, `j` inchangé)
        * `Y` : gap **horizontal** (avance en `j`, `i` inchangé)
      Avec réinitialisation à 0 autorisée (local).
    - Traceback : démarre à la meilleure cellule globale sur {M, X, Y}
      et s'arrête quand un état vaut 0.
    - Complexité : O(P·Q) en temps et mémoire.
    """
    P, Q = L.shape if L.size else (0, 0)
    if P == 0 or Q == 0:
        return 0.0, []

    # Matrices de DP (local : max(..., 0))
    M = np.zeros((P + 1, Q + 1), dtype=np.float64)  # fin par match (consomme L)
    X = np.zeros((P + 1, Q + 1), dtype=np.float64)  # gap vertical (avance i)
    Y = np.zeros((P + 1, Q + 1), dtype=np.float64)  # gap horizontal (avance j)

    # Traceback: 0=stop, 1=M, 2=X, 3=Y
    TB_M = np.zeros((P + 1, Q + 1), dtype=np.uint8)
    TB_X = np.zeros((P + 1, Q + 1), dtype=np.uint8)
    TB_Y = np.zeros((P + 1, Q + 1), dtype=np.uint8)

    best_score = 0.0
    best_cell  = (0, 0, 1)  # (i, j, state) — state ∈ {1(M), 2(X), 3(Y)}

    go, ge = float(gap_open), float(gap_ext)

    for i in range(1, P + 1):
        for j in range(1, Q + 1):
            s = float(L[i - 1, j - 1])

            # X : gap vertical (ouvre/étend)
            x_from_M = M[i - 1, j] - go - ge
            x_from_X = X[i - 1, j] - ge
            X[i, j] = max(0.0, x_from_M, x_from_X)
            TB_X[i, j] = 0 if X[i, j] == 0 else (1 if X[i, j] == x_from_M else 2)

            # Y : gap horizontal (ouvre/étend)
            y_from_M = M[i, j - 1] - go - ge
            y_from_Y = Y[i, j - 1] - ge
            Y[i, j] = max(0.0, y_from_M, y_from_Y)
            TB_Y[i, j] = 0 if Y[i, j] == 0 else (1 if Y[i, j] == y_from_M else 3)

            # M : consomme L[i-1, j-1]
            m_from_M = M[i - 1, j - 1] + s
            m_from_X = X[i - 1, j - 1] + s
            m_from_Y = Y[i - 1, j - 1] + s
            M[i, j]  = max(0.0, m_from_M, m_from_X, m_from_Y)
            TB_M[i, j] = 0 if M[i, j] == 0 else (1 if M[i, j] == m_from_M
                                                  else (2 if M[i, j] == m_from_X else 3))

            # Meilleure cellule globale sur {M, X, Y}
            if M[i, j] >= best_score:
                best_score = M[i, j]; best_cell = (i, j, 1)
            if X[i, j] >= best_score:
                best_score = X[i, j]; best_cell = (i, j, 2)
            if Y[i, j] >= best_score:
                best_score = Y[i, j]; best_cell = (i, j, 3)

    # Traceback local : depuis best_cell, jusqu'à 0
    path: list[tuple[int, int]] = []
    i, j, state = best_cell
    while i > 0 and j > 0 and best_score > 0:
        if state == 1:  # M
            if TB_M[i, j] == 0:
                break
            path.append((i - 1, j - 1))      # correspond à L[i-1, j-1]
            prev = TB_M[i, j]
            i, j, state = i - 1, j - 1, prev
        elif state == 2:  # X (gap vertical)
            if TB_X[i, j] == 0:
                break
            prev = TB_X[i, j]
            i, j, state = i - 1, j, (1 if prev == 1 else 2)
        else:  # state == 3, Y (gap horizontal)
            if TB_Y[i, j] == 0:
                break
            prev = TB_Y[i, j]
            i, j, state = i, j - 1, (1 if prev == 1 else 3)

    path.reverse()
    return float(best_score), path


def score_local(dope: np.ndarray,
                D_tmpl: np.ndarray,
                seq_idx: np.ndarray,
                m: int, n: int,
                q_window: int | None = DEFAULT_Q_WINDOW,
                cutoff: float = PAIRWISE_CUTOFF,
                gap_open: float = GAP_OPEN_LOCAL,
                gap_ext: float  = GAP_EXT_LOCAL) -> tuple[float, dict]:
    """
    Enveloppe pour `(m, n)` :
    construit la matrice d'environnement **L**, lance Smith-Waterman local,
    et renvoie le score `Q(m,n)` ainsi que des **informations de debug**.

    Parameters
    ----------
    dope : numpy.ndarray, shape (20, 20, N_BINS)
        Tenseur DOPE Cα-Cα.
    D_tmpl : numpy.ndarray, shape (M, M)
        Matrice de distances du gabarit.
    seq_idx : numpy.ndarray, shape (N,)
        Séquence encodée (indices 0..19).
    m : int
        Index gabarit (0..M-1).
    n : int
        Index séquence (0..N-1).
    q_window : int or None, optional
        Demi-fenêtre autour de `n` (None = tous `q`).
    cutoff : float, optional
        Cutoff de distance (Å) pour contrib. DOPE.
    gap_open : float, optional
        Coût d'ouverture de gap pour SW local.
    gap_ext : float, optional
        Coût d'extension de gap pour SW local.

    Returns
    -------
    score : float
        `Q(m, n)` = meilleur score local.
    info : dict
        Détails utiles au debug :
        - "L_shape": tuple (|P|, |Q|)
        - "n_neighbors": |P|
        - "n_q": |Q|
        - "path_len": longueur du chemin local
        - "P_list": indices réels `p` côté gabarit
        - "Q_list": indices réels `q` côté séquence
        - "path": chemin dans les indices **de L** (ip, jq)

    Notes
    -----
    Biologie : `Q(m,n)` mesure la compatibilité **environnementale** 3D du résidu
    `m` (gabarit) avec des positions de la séquence autour de `n`. Informatique :
    c'est la brique “local DP” qui alimente ensuite l'alignement **global**.
    """
    L, P_list, Q_list = lowlevel_env_matrix(
        dope, D_tmpl, seq_idx, m, n, q_window=q_window, cutoff=cutoff
    )
    score, path = smith_waterman_local(L, gap_open=gap_open, gap_ext=gap_ext)
    info = {
        "L_shape": L.shape,
        "n_neighbors": len(P_list),
        "n_q": len(Q_list),
        "path_len": len(path),
        "P_list": P_list,
        "Q_list": Q_list,
        "path": path,
    }
    return score, info



# ==============================================
# 8 — Sélection initiale & construction Q
# ==============================================

def initial_pair_selection(seq_idx: np.ndarray,
                           D_tmpl: np.ndarray,
                           sse: list[str] | None = None,
                           K0: int = K0_SELECTION) -> list[tuple[int, int]]:
    """
    Construire une **sélection initiale** de paires (m, n) plausibles.

    Stratégie
    ---------
    Échantillonnage quasi-uniforme **le long de la diagonale proportionnelle** :
    n ≈ m · (N - 1) / (M - 1), afin de couvrir tout le domaine (M×N) avec K0
    ancres réparties. Cette graine est **sobre** et **robuste** (peu de biais).

    Parameters
    ----------
    seq_idx : numpy.ndarray, shape (N,)
        Séquence cible encodée (indices 0..19).
    D_tmpl : numpy.ndarray, shape (M, M)
        Matrice des distances Cα-Cα du gabarit (seule la taille M est utilisée ici).
    sse : list of str or None, optional
        Éventuelle annotation SSE (non utilisée dans cette version simple).
    K0 : int, optional
        Nombre d'ancres initiales à placer.

    Returns
    -------
    list of (int, int)
        Liste de paires `(m, n)` (sans doublons), de longueur ≤ K0.

    Notes
    -----
    - Biologie : on suppose un **étirement global** raisonnable entre gabarit
      et séquence (diagonale). Les itérations affineront ensuite localement.
    - Informatique : cette sélection ne dépend **pas** du contenu DOPE, donc
      elle ne “sur-apprend” pas dès le départ.
    """
    M = D_tmpl.shape[0]
    N = int(seq_idx.shape[0])
    if M == 0 or N == 0:
        return []

    pairs: list[tuple[int, int]] = []
    # Positions centrées sur K0 quanta le long de la diagonale proportionnelle
    for t in range(K0):
        m = int(round((t + 0.5) * (M / K0))) - 1
        m = min(max(m, 0), M - 1)
        # Projection proportionnelle
        n = int(round(m * (N - 1) / (M - 1))) if M > 1 else 0
        n = min(max(n, 0), N - 1)
        if (m, n) not in pairs:
            pairs.append((m, n))

    return pairs


def compute_Q_matrix(dope: np.ndarray,
                     D_tmpl: np.ndarray,
                     seq_idx: np.ndarray,
                     selection: list[tuple[int, int]],
                     q_window: int = DEFAULT_Q_WINDOW,
                     cutoff: float = PAIRWISE_CUTOFF) -> tuple[np.ndarray, dict]:
    """
    Construire la matrice **Q[M, N]** : score local `Q(m, n)` pour les paires sélectionnées.

    Pour chaque `(m, n)` de `selection`, on :
      1) construit la matrice d'environnement **L** (voisins de `m` vs fenêtres autour de `n`),
      2) exécute **Smith-Waterman local** (affine) sur `L`,
      3) affecte `Q[m, n] ← score_local`.

    Toutes les autres cellules restent à **0** (ne sont pas évaluées à ce stade).

    Parameters
    ----------
    dope : numpy.ndarray, shape (20, 20, N_BINS)
        Tenseur DOPE Cα-Cα.
    D_tmpl : numpy.ndarray, shape (M, M)
        Matrice des distances Cα-Cα du gabarit.
    seq_idx : numpy.ndarray, shape (N,)
        Séquence cible encodée (indices 0..19).
    selection : list of (int, int)
        Paires `(m, n)` à évaluer (voir `initial_pair_selection`).
    q_window : int, optional
        Demi-fenêtre autour de `n` pour la construction de L.
    cutoff : float, optional
        Cutoff de distance (Å) pour les contributions DOPE.

    Returns
    -------
    Q : numpy.ndarray, shape (M, N), dtype float64
        Matrice partiellement remplie (0 ailleurs) des scores locaux.
    info : dict
        Résumé pour logs/diagnostic :
        - "n_evaluated": nombre de paires évaluées
        - "top5": top-5 des (m, n, score, path_len, |P|, |Q|)
        - "mean_score": moyenne des scores calculés

    Notes
    -----
    - Biologie : `Q(m, n)` capture la **compatibilité d'environnement** entre un
      résidu gabarit `m` et une position de séquence `n`.
    - Informatique : `Q` servira de base à `H` pour l'alignement global (NW/Gotoh),
      puis sera **enrichi** itérativement (sélection/renforcement).
    """
    M = D_tmpl.shape[0]
    N = int(seq_idx.shape[0])
    Q = np.zeros((M, N), dtype=np.float64)

    logs = []
    for (m, n) in selection:
        sc, meta = score_local(dope, D_tmpl, seq_idx, m, n,
                               q_window=q_window, cutoff=cutoff)
        Q[m, n] = sc
        logs.append((m, n, sc, meta["path_len"], meta["n_neighbors"], meta["n_q"]))

    # Petit résumé (pour le logging)
    top5 = sorted(logs, key=lambda x: x[2], reverse=True)[:5] if logs else []

    info = {
        "n_evaluated": len(selection),
        "top5": top5,  # (m, n, score, path_len, |P|, |Q|)
        "mean_score": float(np.mean([x[2] for x in logs])) if logs else 0.0,
    }
    return Q, info



# ==============================================
# Renforcement SAP-like de H
# ==============================================

def compute_Q_and_sapR(dope: np.ndarray,
                        D_tmpl: np.ndarray,
                        seq_idx: np.ndarray,
                        selection: set[tuple[int, int]],
                        q_window: int = DEFAULT_Q_WINDOW,
                        cutoff: float = PAIRWISE_CUTOFF,
                        sap_norm: str = SAP_NORM) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Calculer simultanément :
      1) la matrice **Q** (scores locaux `Q(m,n)` sur les paires sélectionnées),
      2) la matrice de **renforcement** **R** de type SAP, en “réémettant”
         le score le long du **chemin local** (p,q) issu de Smith-Waterman.

    Idée (SAP-like)
    ---------------
    Pour chaque hypothèse `(m ↔ n)` :
      - On construit la matrice d'environnement `L` (voisins `p` de `m` vs positions `q` autour de `n`).
      - On exécute **SW local** pour obtenir un chemin optimal `(ip, iq)` dans `L`.
      - Pour chaque pas du chemin, on identifie les indices **réels** `(p_real, q_real)` et
        on renforce `R[p_real, q_real]` par une contribution **positive** proportionnelle
        à `max(L[ip, iq], 0)` (optionnellement normalisée par la longueur du chemin).

    Parameters
    ----------
    dope : numpy.ndarray, shape (20, 20, N_BINS)
        Tenseur DOPE Cα-Cα (énergies). On travaille en score `= -énergie`.
    D_tmpl : numpy.ndarray, shape (M, M)
        Matrice des distances Cα-Cα du gabarit (template).
    seq_idx : numpy.ndarray, shape (N,)
        Séquence cible encodée 0..19.
    selection : set[tuple[int, int]]
        Ensemble de paires `(m, n)` à évaluer (sélection courante).
    q_window : int, optional
        Demi-fenêtre autour de `n` pour restreindre les `q` dans `L`.
    cutoff : float, optional
        Cutoff distance (Å) pour les contributions DOPE.
    sap_norm : {"length","none"}, optional
        Normalisation de la contribution le long du chemin :
        - "length" : divise chaque contribution par `len(path)` (rend R comparable entre chemins).
        - "none"   : pas de normalisation.

    Returns
    -------
    Q : numpy.ndarray, shape (M, N), dtype float64
        Scores locaux `Q(m,n)` **uniquement** sur les `(m,n)` de `selection` (0 ailleurs).
    R : numpy.ndarray, shape (M, N), dtype float64
        Matrice de renforcement cumulée (≥ 0), alimentée par les chemins locaux.
    info : dict
        Résumé pour logs/diagnostic :
        - "n_evaluated": nombre de paires évaluées
        - "top5": top-5 (m, n, score, path_len, |P|, |Q|)
        - "mean_score": moyenne des `Q(m,n)` calculés

    Notes
    -----
    - **Biologie** : `R` met en avant des **crêtes cohérentes** où l'environnement
      structural de `m` “trouve” localement une correspondance régulière dans la séquence.
    - **Math/Algo** : `R[p,q] += max(L[ip,iq],0) / denom` pour chaque pas du chemin,
      où `denom = len(path)` si `sap_norm == "length"` sinon `denom = 1`.
      On évite d'ajouter des contributions négatives (clamp à 0).
    - `Q` et `R` seront combinés plus tard dans `H = Q + α·Bias + β·R` (si activé).

    """
    M = D_tmpl.shape[0]
    N = int(seq_idx.shape[0])
    Q = np.zeros((M, N), dtype=np.float64)
    R = np.zeros((M, N), dtype=np.float64)

    logs = []
    for (m, n) in selection:
        # 1) Matrice d'environnement L pour l'hypothèse (m, n)
        L, P_list, Q_list = lowlevel_env_matrix(
            dope, D_tmpl, seq_idx, m, n, q_window=q_window, cutoff=cutoff
        )

        # 2) Score local et chemin dans L
        sc, path = smith_waterman_local(L, gap_open=GAP_OPEN_LOCAL, gap_ext=GAP_EXT_LOCAL)
        Q[m, n] = sc
        logs.append((m, n, sc, len(path), len(P_list), len(Q_list)))

        if len(path) == 0 or L.size == 0:
            continue  # rien à renforcer si pas de chemin (ou L vide)

        # 3) Réémission des contributions positives le long du chemin
        denom = float(len(path)) if (sap_norm == "length" and len(path) > 0) else 1.0
        for (ip, iq) in path:
            p_real = P_list[ip]
            q_real = Q_list[iq]
            contrib = float(max(L[ip, iq], 0.0)) / denom  # clamp négatif à 0
            R[p_real, q_real] += contrib

    info = {
        "n_evaluated": len(selection),
        "top5": sorted(logs, key=lambda x: x[2], reverse=True)[:5],
        "mean_score": float(np.mean([x[2] for x in logs])) if logs else 0.0,
    }
    return Q, R, info



# ==============================================
# 9 — Needleman-Wunsch/Gotoh global (H/Q → threading)
# ==============================================

def _vertical_gap_costs_by_row(sse: list[str] | None,
                               M: int,
                               go_loop: float = GAP_OPEN_V_LOOP,
                               ge_loop: float = GAP_EXT_V_LOOP,
                               go_sse: float  = GAP_OPEN_V_SSE,
                               ge_sse: float  = GAP_EXT_V_SSE) -> tuple[np.ndarray, np.ndarray]:
    """
    Construire des coûts de gaps **verticaux** dépendant de la SSE pour chaque ligne i.

    Parameters
    ----------
    sse : list[str] or None
        Annotations de structure secondaire pour le gabarit (longueur ≥ M),
        avec des labels parmi {"H","E","C"} (Hélice, brin (beta), Coil).
        Si None, on considère tout comme **boucle/coil**.
    M : int
        Nombre de résidus dans le gabarit (taille verticale).
    go_loop, ge_loop : float
        Coûts d'ouverture/extension de **gap vertical** en boucle/coil.
    go_sse, ge_sse : float
        Coûts d'ouverture/extension de **gap vertical** en SSE (H/E), plus sévères.

    Returns
    -------
    goV : numpy.ndarray, shape (M,), dtype float64
        Coûts d'ouverture de gap vertical par ligne i.
    geV : numpy.ndarray, shape (M,), dtype float64
        Coûts d'extension de gap vertical par ligne i.

    Notes
    -----
    - **Vertical = suppression dans la structure** (on consomme `i`, pas `j`).
      Biologiquement : on évite d'ouvrir des gaps au milieu d'un **hélice ou brin**,
      donc pénalité plus forte en H/E.
    """
    goV = np.empty(M, dtype=np.float64)
    geV = np.empty(M, dtype=np.float64)
    if sse is None:
        goV[:] = go_loop
        geV[:] = ge_loop
        return goV, geV

    for i in range(M):
        lab = sse[i] if i < len(sse) and sse[i] is not None else "C"
        if lab in ("H", "E"):  # hélice / brin
            goV[i], geV[i] = go_sse, ge_sse
        else:                  # boucle / coil
            goV[i], geV[i] = go_loop, ge_loop
    return goV, geV


def nw_gotoh_global(H: np.ndarray,
                    sse: list[str] | None = None,
                    gap_open_h: float = GAP_OPEN_H,
                    gap_ext_h:  float = GAP_EXT_H,
                    go_v_loop: float = GAP_OPEN_V_LOOP,
                    ge_v_loop: float = GAP_EXT_V_LOOP,
                    go_v_sse:  float = GAP_OPEN_V_SSE,
                    ge_v_sse:  float = GAP_EXT_V_SSE) -> tuple[float, list[tuple[int | None, int | None]]]:
    """
    Needleman-Wunsch **global** avec pénalités **affines** (Gotoh) sur la matrice `H`.

    Parameters
    ----------
    H : numpy.ndarray, shape (M, N)
        Matrice de **scores à maximiser** (ex. `H = Q` ou `Q + β·R + α·Bias`).
    sse : list[str] or None, optional
        Annotations SSE du gabarit (labels 'H','E','C') pour moduler les **gaps verticaux**.
        Si None, coûts verticaux uniformes (mode boucle).
    gap_open_h, gap_ext_h : float, optional
        Coûts d'ouverture/extension pour **gaps horizontaux** (dans la séquence).
    go_v_loop, ge_v_loop : float, optional
        Coûts d'ouverture/extension pour **gaps verticaux** en boucle/coil.
    go_v_sse, ge_v_sse : float, optional
        Coûts d'ouverture/extension pour **gaps verticaux** en SSE (H/E), plus sévères.

    Returns
    -------
    best_score : float
        Score global optimal.
    path : list[tuple[int | None, int | None]]
        Chemin d'alignement global (du début à la fin), où :
          - `(i, j)`  : match/mismatch (consomme H[i, j])
          - `(i, None)`: **gap horizontal** (insertion dans la séquence → on consomme `i` côté structure)
          - `(None, j)`: **gap vertical**   (suppression côté structure → on consomme `j` côté séquence)

        NB : par convention dans ce code, on logge `(i, None)` pour un gap **horizontal**
        (on avance dans `i`) et `(None, j)` pour un gap **vertical** (on avance dans `j`).
        C'est cohérent avec l'implémentation X/Y ci-dessous.

    Notes
    -----
    - Trois matrices Gotoh :
        * `Mx` : fin par match/mismatch (consomme `H[i-1, j-1]`)
        * `X`  : fin par **gap vertical** (consomme `i`, `j` inchangé)
        * `Y`  : fin par **gap horizontal** (consomme `j`, `i` inchangé)
      et trois tables de traceback associées.
    - Initialisations **globales** : première ligne/colonne = chaînes de gaps.
    - Complexité : O(M·N) en temps et mémoire.
    """
    M, N = H.shape
    NEG_INF = -1e300

    # Coûts verticaux dépendants de la SSE (par ligne i)
    goV, geV = _vertical_gap_costs_by_row(sse, M, go_v_loop, ge_v_loop, go_v_sse, ge_v_sse)
    goH, geH = float(gap_open_h), float(gap_ext_h)

    # DP : Mx (match), X (gap vertical), Y (gap horizontal)
    Mx = np.full((M + 1, N + 1), NEG_INF, dtype=np.float64)
    X  = np.full((M + 1, N + 1), NEG_INF, dtype=np.float64)
    Y  = np.full((M + 1, N + 1), NEG_INF, dtype=np.float64)

    # Traceback: 0=Mx, 1=X, 2=Y — on stocke l'état d'origine
    TB_M = np.zeros((M + 1, N + 1), dtype=np.uint8)
    TB_X = np.zeros((M + 1, N + 1), dtype=np.uint8)
    TB_Y = np.zeros((M + 1, N + 1), dtype=np.uint8)

    # Init global
    Mx[0, 0] = 0.0
    X[0, 0]  = NEG_INF
    Y[0, 0]  = NEG_INF

    # Première colonne (j=0) : colonnes vides → gaps verticaux successifs (avance i)
    for i in range(1, M + 1):
        open_cost   = Mx[i - 1, 0] - goV[i - 1] - geV[i - 1]
        extend_cost = X[i - 1, 0]  - geV[i - 1]
        X[i, 0]   = max(open_cost, extend_cost)
        TB_X[i, 0] = 0 if X[i, 0] == open_cost else 1
        Mx[i, 0] = NEG_INF
        Y[i, 0]  = NEG_INF

    # Première ligne (i=0) : lignes vides → gaps horizontaux successifs (avance j)
    for j in range(1, N + 1):
        open_cost   = Mx[0, j - 1] - goH - geH
        extend_cost = Y[0, j - 1]  - geH
        Y[0, j]   = max(open_cost, extend_cost)
        TB_Y[0, j] = 0 if Y[0, j] == open_cost else 2
        Mx[0, j] = NEG_INF
        X[0, j]  = NEG_INF

    # Remplissage
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            s = float(H[i - 1, j - 1])

            # X : gap vertical se terminant à (i, j) → on consomme i
            x_from_M = Mx[i - 1, j] - goV[i - 1] - geV[i - 1]
            x_from_X = X[i - 1, j]  - geV[i - 1]
            if x_from_M >= x_from_X:
                X[i, j] = x_from_M; TB_X[i, j] = 0  # depuis Mx
            else:
                X[i, j] = x_from_X; TB_X[i, j] = 1  # depuis X

            # Y : gap horizontal se terminant à (i, j) → on consomme j
            y_from_M = Mx[i, j - 1] - goH - geH
            y_from_Y = Y[i, j - 1]  - geH
            if y_from_M >= y_from_Y:
                Y[i, j] = y_from_M; TB_Y[i, j] = 0  # depuis Mx
            else:
                Y[i, j] = y_from_Y; TB_Y[i, j] = 2  # depuis Y

            # Mx : match/mismatch (consomme H[i-1, j-1])
            m_from_M = Mx[i - 1, j - 1] + s
            m_from_X = X[i - 1, j - 1]  + s
            m_from_Y = Y[i - 1, j - 1]  + s
            if m_from_M >= m_from_X and m_from_M >= m_from_Y:
                Mx[i, j] = m_from_M; TB_M[i, j] = 0  # from Mx
            elif m_from_X >= m_from_Y:
                Mx[i, j] = m_from_X; TB_M[i, j] = 1  # from X
            else:
                Mx[i, j] = m_from_Y; TB_M[i, j] = 2  # from Y

    # Score global = meilleur des trois à (M, N)
    candidates = [(Mx[M, N], 0), (X[M, N], 1), (Y[M, N], 2)]
    best_score, state = max(candidates, key=lambda t: t[0])

    # Traceback global (jusqu'à (0,0))
    path: list[tuple[int | None, int | None]] = []
    i, j = M, N
    while i > 0 or j > 0:
        if state == 0:  # Mx : match
            prev = TB_M[i, j]
            path.append((i - 1, j - 1))      # match réel (i-1, j-1)
            i, j, state = i - 1, j - 1, prev
        elif state == 1:  # X : gap vertical (on consomme i)
            prev = TB_X[i, j]
            path.append((i - 1, None))
            i, j, state = i - 1, j, (0 if prev == 0 else 1)
        else:  # state == 2 : Y : gap horizontal (on consomme j)
            prev = TB_Y[i, j]
            path.append((None, j - 1))
            i, j, state = i, j - 1, (0 if prev == 0 else 2)

    path.reverse()
    return float(best_score), path



# ==============================================
# 10 — Orchestrateur sélection/itération
# ==============================================

def _normalize_matrix(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normaliser linéairement une matrice dans [0, 1].

    Parameters
    ----------
    X : numpy.ndarray
        Matrice de scores.
    eps : float, optional
        Petit seuil pour éviter une division par ~0 lorsque X est constante.

    Returns
    -------
    numpy.ndarray
        (X - min) / (max - min) si la dynamique est suffisante, sinon matrice de zéros.

    Notes
    -----
    Utilisé pour créer un **biais** "souvenir" borné (entre 0 et 1) qui sera
    accumulé d'un cycle à l'autre : `Bias <- γ·Bias + (1-γ)·norm(H)`.
    """
    x_min = float(np.min(X))
    x_max = float(np.max(X))
    if x_max - x_min < eps:
        return np.zeros_like(X, dtype=np.float64)
    return (X - x_min) / (x_max - x_min)


def _path_matches_only(aln_path: list[tuple[int | None, int | None]]) -> list[tuple[int, int]]:
    """
    Extraire uniquement les **matches** (i, j) d'un chemin global.

    Parameters
    ----------
    aln_path : list of tuple(int|None, int|None)
        Chemin d'alignement global (NW/Gotoh), mixant matches et gaps.

    Returns
    -------
    list of (int, int)
        Couples (i, j) où i et j sont définis (positions alignées).
    """
    return [(i, j) for (i, j) in aln_path if i is not None and j is not None]


def _selection_metrics(selection: set[tuple[int, int]],
                       aln_pairs: set[tuple[int, int]]) -> tuple[float, float]:
    """
    Mesurer la cohérence entre la **sélection courante** et le **chemin global**.

    Parameters
    ----------
    selection : set of (int, int)
        Paires (m, n) actuellement évaluées (graines + ajouts).
    aln_pairs : set of (int, int)
        Paires (i, j) issues du chemin d'alignement (matches uniquement).

    Returns
    -------
    sel_on_aln : float
        Proportion des paires de la **sélection** qui tombent sur le chemin.
    aln_in_sel : float
        Proportion du **chemin** qui est couverte par la sélection.

    Notes
    -----
    Ces métriques servent à évaluer la **convergence** : si le chemin tombe
    majoritairement dans la zone déjà sélectionnée (et que le score n'augmente
    plus), on peut arrêter.
    """
    if not selection or not aln_pairs:
        return 0.0, 0.0
    inter = selection & aln_pairs
    sel_on_aln = len(inter) / len(selection)
    aln_in_sel = len(inter) / len(aln_pairs)
    return float(sel_on_aln), float(aln_in_sel)


def _expand_selection_from_path(M: int, N: int,
                                aln_pairs: list[tuple[int, int]],
                                win: int = ALN_NEIGH_WIN) -> set[tuple[int, int]]:
    """
    Élargir la sélection autour du **chemin global** en ajoutant un voisinage.

    Parameters
    ----------
    M, N : int
        Dimensions de la matrice (gabarit x séquence).
    aln_pairs : list of (int, int)
        Liste des matches (i, j) du chemin.
    win : int, optional
        Demi-fenêtre (carrée) autour de chaque (i, j) à ajouter.

    Returns
    -------
    set of (int, int)
        Nouvelles paires à ajouter à la sélection.

    Notes
    -----
    Biologie : on “balaie” localement autour du chemin pour capter des
    alternatives proches (glissements d'un ou deux résidus).
    """
    new_sel: set[tuple[int, int]] = set()
    for (i, j) in aln_pairs:
        i0, i1 = max(0, i - win), min(M - 1, i + win)
        j0, j1 = max(0, j - win), min(N - 1, j + win)
        for ii in range(i0, i1 + 1):
            for jj in range(j0, j1 + 1):
                new_sel.add((ii, jj))
    return new_sel


def _topk_peaks(H: np.ndarray,
                k: int,
                banned: set[tuple[int, int]] | None = None) -> list[tuple[int, int]]:
    """
    Sélectionner les `k` meilleurs **pics** de `H` (scores max), en excluant `banned`.

    Parameters
    ----------
    H : numpy.ndarray, shape (M, N)
        Matrice de scores.
    k : int
        Nombre de positions (i, j) à retourner.
    banned : set of (int, int) or None
        Ensemble de positions à ignorer (ex. déjà sélectionnées ou voisinage du chemin).

    Returns
    -------
    list of (int, int)
        Meilleures positions (i, j) triées grossièrement par score décroissant.
    """
    M, N = H.shape
    flat = H.ravel()
    order = np.argsort(-flat)  # indices triés par score décroissant
    peaks: list[tuple[int, int]] = []
    banned = banned or set()
    for idx in order:
        if len(peaks) >= k:
            break
        i, j = divmod(int(idx), N)
        if (i, j) in banned:
            continue
        peaks.append((i, j))
    return peaks


def compute_Q_on_selection(dope: np.ndarray,
                           D_tmpl: np.ndarray,
                           seq_idx: np.ndarray,
                           selection: set[tuple[int, int]],
                           q_window: int = DEFAULT_Q_WINDOW,
                           cutoff: float = PAIRWISE_CUTOFF) -> np.ndarray:
    """
    Calculer `Q` **uniquement** sur la sélection courante.

    Parameters
    ----------
    dope : numpy.ndarray
        Tenseur DOPE (20×20×N_BINS).
    D_tmpl : numpy.ndarray
        Matrice de distances Cα-Cα du gabarit.
    seq_idx : numpy.ndarray
        Séquence encodée (0..19).
    selection : set of (int, int)
        Paires `(m, n)` à évaluer.
    q_window : int, optional
        Demi-fenêtre autour de `n` pour la matrice locale.
    cutoff : float, optional
        Cutoff de distance (Å) pour les contributions DOPE.

    Returns
    -------
    numpy.ndarray
        Matrice `Q` (M×N) partiellement remplie (0 ailleurs).
    """
    M = D_tmpl.shape[0]
    N = int(seq_idx.shape[0])
    Q = np.zeros((M, N), dtype=np.float64)

    for (m, n) in selection:
        sc, _ = score_local(dope, D_tmpl, seq_idx, m, n,
                            q_window=q_window, cutoff=cutoff)
        Q[m, n] = sc
    return Q


def iterate_ddp(dope: np.ndarray,
                D_tmpl: np.ndarray,
                seq_idx: np.ndarray,
                sse: list[str] | None = None,
                K0: int = K0_SELECTION,
                cycles: int = N_CYCLES,
                expand_k: int = EXPAND_K,
                q_window: int = DEFAULT_Q_WINDOW,
                cutoff: float = PAIRWISE_CUTOFF,
                alpha_bias: float = ALPHA_BIAS,
                gamma_bias: float = GAMMA_BIAS,
                use_sap: bool = False,
                beta_sap: float = SAP_BETA,
                sap_norm: str = SAP_NORM) -> tuple[list[tuple[int | None, int | None]], float, np.ndarray]:
    """
    Boucle **itérative** de threading par double programmation dynamique (DDP).

    Pipeline (par cycle)
    --------------------
    1) **Local** : (au choix)
       - Sans SAP → `Q ← compute_Q_on_selection(...)`
       - Avec SAP → `Q, R ← compute_Q_and_sapR(...)`
    2) **Combinaison** : `H ← Q + α·Bias (+ β·R si SAP)`
    3) **Global** : `aln_path, best_score ← nw_gotoh_global(H, sse)`
    4) **Sélection** :
       - Extraire les matches `(i, j)` du chemin.
       - Élargir autour du chemin (`_expand_selection_from_path`).
       - Ajouter `k` nouveaux **pics** de `H` hors zone déjà couverte.
    5) **Mémoire** :
       - `Bias ← γ·Bias + (1−γ)·normalize(H)`  (souvenir borné des hotspots)
    6) **Arrêt** :
       - Si `Δscore < MIN_IMPROV` **et** si le chemin recouvre suffisamment la sélection,
         on s'arrête et on renvoie le meilleur alignement.

    Biologie
    --------
    Le processus alterne **compatibilité locale d'environnement** (SW sur L)
    et **cohérence globale** (NW sur H). Le renforcement de type SAP construit
    des **crêtes** là où les environnements locaux s'enchaînent de façon régulière.

    Parameters
    ----------
    dope, D_tmpl, seq_idx : voir fonctions précédentes
    sse : list[str] or None, optional
        SSE du gabarit pour moduler les gaps **verticaux** lors du global.
    K0 : int, optional
        Taille de la sélection initiale (graines le long de la diagonale).
    cycles : int, optional
        Nombre maximum de cycles.
    expand_k : int, optional
        Nombre de **pics** supplémentaires de `H` ajoutés par cycle.
    q_window, cutoff : see above
    alpha_bias : float, optional
        Poids `α` du biais accumulé (mémoire).
    gamma_bias : float, optional
        Facteur d'oubli `γ` du biais (0 = remplace, 1 = fige).
    use_sap : bool, optional
        Activer le renforcement SAP-like (accumulation sur `R`).
    beta_sap : float, optional
        Poids `β` de `R` dans `H`.
    sap_norm : {"length","none"}, optional
        Normalisation de `R` (voir `compute_Q_and_sapR`).

    Returns
    -------
    aln_path : list[tuple[int|None, int|None]]
        Chemin d'alignement global final.
    best_score : float
        Score global associé.
    H : numpy.ndarray
        Dernière matrice `H` (pour inspection/sauvegarde).

    Notes
    -----
    - Critère d'arrêt : `Δscore < MIN_IMPROV` **et** couverture suffisante
      (`aln_in_sel ≥ 0.6` ou `sel_on_aln ≥ 0.6`). Cela évite de s'arrêter
      trop tôt sur un plateau bruité.
    - Les expansions de sélection limitent le **piégeage local** en explorant
      autour du chemin et en ajoutant des crêtes fortes de `H`.
    """
    M = D_tmpl.shape[0]; N = int(seq_idx.shape[0])

    # 0) Graines initiales
    init_pairs = initial_pair_selection(seq_idx, D_tmpl, sse=None, K0=K0)
    selection: set[tuple[int, int]] = set(init_pairs)

    # Biais (souvenir borné) et meilleur score précédent
    Bias = np.zeros((M, N), dtype=np.float64)
    prev_score = -1e300

    for k in range(1, cycles + 1):
        # 1) Local : Q (et éventuellement R)
        if use_sap:
            Q, R, info = compute_Q_and_sapR(
                dope, D_tmpl, seq_idx, selection,
                q_window=q_window, cutoff=cutoff, sap_norm=sap_norm
            )
        else:
            Q = compute_Q_on_selection(
                dope, D_tmpl, seq_idx, selection,
                q_window=q_window, cutoff=cutoff
            )
            R = np.zeros_like(Q)
            info = {
                "n_evaluated": len(selection),
                "top5": [],
                "mean_score": float(np.mean(Q[Q != 0])) if np.any(Q) else 0.0,
            }

        # 2) Combinaison → H
        H = Q + alpha_bias * Bias + (beta_sap * R if use_sap else 0.0)

        # 3) Global : chemin et score
        best_score, aln_path = nw_gotoh_global(H, sse=sse)
        aln_pairs = set(_path_matches_only(aln_path))

        # 4) Métriques de couverture (diagnostic / convergence)
        sel_on_aln, aln_in_sel = _selection_metrics(selection, aln_pairs)
        LOGGER.info(
            "[Cycle %d%s] |S|=%d ; score=%.3f ; mean(Q)=%.3f ; %%sel_on_aln=%.1f%% ; %%aln_in_sel=%.1f%%",
            k, " +SAP" if use_sap else "", len(selection), best_score, info["mean_score"],
            100 * sel_on_aln, 100 * aln_in_sel
        )

        # 5) Expansion de la sélection : voisinage du chemin + pics de H
        sel_from_path = _expand_selection_from_path(M, N, list(aln_pairs), win=ALN_NEIGH_WIN)
        banned = selection | sel_from_path
        peaks = _topk_peaks(H, k=expand_k, banned=banned)
        selection |= sel_from_path
        selection |= set(peaks)

        # 6) Mémoire : mise à jour du biais borné
        Bias = gamma_bias * Bias + (1.0 - gamma_bias) * _normalize_matrix(H)

        # 7) Arrêt si amélioration marginale et bonne couverture
        score_improv = best_score - prev_score
        if (score_improv < MIN_IMPROV) and (aln_in_sel >= 0.6 or sel_on_aln >= 0.6):
            LOGGER.info("[Cycle %d] Convergence atteinte (Δscore=%.3g).", k, score_improv)
            return aln_path, best_score, H

        prev_score = best_score

    LOGGER.info("Itérations terminées (max cycles=%d).", cycles)
    return aln_path, best_score, H



# ==============================================
# 11 — Énergie finale & z-score
# ==============================================

def _alignment_matches_only(aln_path: list[tuple[int | None, int | None]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Extraire les **matches** (i, j) d'un chemin global et les retourner
    sous forme de deux tableaux parallèles.

    Parameters
    ----------
    aln_path : list of tuple(int|None, int|None)
        Chemin d'alignement global (ex. NW/Gotoh) contenant des matches et des gaps.

    Returns
    -------
    tmpl_idx : numpy.ndarray of int, shape (L,)
        Indices **gabarit** (i) des positions appariées.
    seq_idx_ : numpy.ndarray of int, shape (L,)
        Indices **séquence** (j) des positions appariées.

    Notes
    -----
    - On ne conserve que les couples où `i` et `j` sont tous deux définis.
    - Utile pour restreindre les calculs d'énergie aux positions **alignées**.
    """
    pairs = [(i, j) for (i, j) in aln_path if i is not None and j is not None]
    if not pairs:
        return np.array([], dtype=int), np.array([], dtype=int)
    tmpl_idx = np.array([p[0] for p in pairs], dtype=int)
    seq_idx_ = np.array([p[1] for p in pairs], dtype=int)
    return tmpl_idx, seq_idx_


def burial_proxy_counts(D_tmpl: np.ndarray, radius: float = BURIAL_RADIUS) -> np.ndarray:
    """
    Compter, pour chaque résidu gabarit, le **nombre de voisins Cα** dans un rayon.

    Parameters
    ----------
    D_tmpl : numpy.ndarray, shape (M, M)
        Matrice de distances Cα-Cα du gabarit.
    radius : float, optional
        Rayon (Å) à l'intérieur duquel on compte les voisins.

    Returns
    -------
    counts : numpy.ndarray of int, shape (M,)
        Nombre de voisins par position (diag exclue).

    Notes
    -----
    - Proxy d'**enterrement** : plus le compte est élevé, plus le résidu est
      vraisemblablement **enfoui** (noyau) plutôt qu'exposé au solvant.
    - Sert si l'on active un terme d'énergie “burial” (pondéré par `W_BURIAL`).
    """
    M = D_tmpl.shape[0]
    mask = (D_tmpl <= float(radius))
    np.fill_diagonal(mask, False)  # ne pas compter soi-même
    return mask.sum(axis=1).astype(np.int32)


def energy_pairwise_dope(dope: np.ndarray,
                         D_tmpl: np.ndarray,
                         seq_idx: np.ndarray,
                         tmpl_idx: np.ndarray,
                         seq_aln_idx: np.ndarray,
                         cutoff: float = PAIRWISE_CUTOFF) -> float:
    """
    Calculer l'**énergie paire-à-paire** (DOPE Cα-Cα) sur les positions **alignées**.

    Définition
    ----------
    Soit `L` le nombre de matches dans l'alignement global. Pour chaque paire
    **ordonnée** `(u, v)` avec `0 ≤ u < v < L` :
      - on récupère `i_u = tmpl_idx[u]`, `i_v = tmpl_idx[v]` (indices gabarit),
      - on prend la distance `d = D_tmpl[i_u, i_v]`,
      - si `d ≤ cutoff`, on ajoute l'énergie DOPE entre les **types AA**
        alignés `aa_u = seq_idx[seq_aln_idx[u]]` et `aa_v = seq_idx[seq_aln_idx[v]]`
        au **bin** correspondant à `d`.

    Paramètres
    ----------
    dope : numpy.ndarray, shape (20, 20, N_BINS)
        Tenseur DOPE (énergies ; plus **petit** = plus favorable).
    D_tmpl : numpy.ndarray, shape (M, M)
        Matrice des distances Cα-Cα du gabarit.
    seq_idx : numpy.ndarray, shape (N,)
        Séquence **complète** encodée (indices 0..19).
    tmpl_idx : numpy.ndarray of int, shape (L,)
        Indices gabarit (i) des positions **alignées** (matches).
    seq_aln_idx : numpy.ndarray of int, shape (L,)
        Indices séquence (j) des positions **alignées** (matches).
    cutoff : float, optional
        Cutoff de distance (Å) au-delà duquel la paire ne contribue pas.

    Returns
    -------
    float
        **Énergie totale** (somme des valeurs DOPE) sur toutes les paires (u < v).
        Valeurs **plus petites** ⇒ alignement/placement plus favorable.

    Notes
    -----
    - Complexité O(L²) (boucle sur toutes les paires alignées).
    - Calcul limité aux **positions alignées** (pas de pénalités gaps ici).
    - On n'inclut que les paires `d ≤ cutoff` (≈ 10 Å), comme en threading classique.
    """
    L = tmpl_idx.shape[0]
    if L <= 1:
        return 0.0

    # Types AA alignés (indices 0..19) dans l'ordre de l'alignement
    aa_aln = seq_idx[seq_aln_idx]

    E = 0.0
    # Boucle O(L^2) sur paires (u < v)
    for u in range(L - 1):
        i_u = int(tmpl_idx[u])
        aa_u = int(aa_aln[u])

        # Vectoriser partiellement sur v > u
        v_range = np.arange(u + 1, L, dtype=int)
        if v_range.size == 0:
            continue

        i_v = tmpl_idx[v_range]
        aa_v = aa_aln[v_range]
        d = D_tmpl[i_u, i_v]  # distances vers tous les v > u

        # Filtrer par cutoff
        ok = d <= cutoff
        if not np.any(ok):
            continue

        dv = d[ok]
        aav = aa_v[ok]

        # Accumuler DOPE(aa_u, aav, bin(dv)) pour chaque paire retenue
        for dvv, aavv in zip(dv, aav):
            k = distance_to_bin(float(dvv))
            E += float(dope[aa_u, int(aavv), k])

    return float(E)


def energy_burial_term(seq_idx: np.ndarray,
                       tmpl_idx: np.ndarray,
                       burial_counts: np.ndarray,
                       w_burial: float = W_BURIAL) -> float:
    """
    Calculer un petit terme optionnel de “solvation / enterrement” :

        E_burial = w_burial * Σ_u [ KD(seq[u]) * (μ - burial_count[i_u]) ]

    où :
      - KD(·) est l'hydropathie **Kyte-Doolittle** (positive = hydrophobe),
      - i_u est l'index gabarit aligné au pas u,
      - μ est la moyenne globale des `burial_counts`.

    Intuition :
      - Un résidu **hydrophobe** (KD>0) peu enterré (burial_count < μ) → pénalité (E_burial ↑).
      - Hydrophobe **bien enterré** (burial_count > μ) → contribution négative (E_burial ↓).

    Parameters
    ----------
    seq_idx : numpy.ndarray of int, shape (L,)
        **Indice AA (0..19) des positions alignées**, dans **l'ordre** du chemin.
        (i.e. ce n'est *pas* la séquence complète, mais le sous-ensemble aligné)
    tmpl_idx : numpy.ndarray of int, shape (L,)
        Indices gabarit alignés (parallèles à `seq_idx`).
    burial_counts : numpy.ndarray of int, shape (M,)
        Nombre de voisins Cα par résidu du gabarit (proxy d'enterrement).
    w_burial : float, optional
        Poids du terme (0.0 = désactivé).

    Returns
    -------
    float
        Énergie d'enterrement. Plus **négatif** = mieux (hydrophobes enterrés).

    Notes
    -----
    - Terme très simple ; par défaut **désactivé** (w_burial=0).
    - Aucune normalisation par L (longueur alignée) ici : tu peux la rajouter
      si tu veux rendre ce terme comparable entre alignements de tailles différentes.
    """
    if w_burial == 0.0 or tmpl_idx.size == 0:
        return 0.0

    mu = float(np.mean(burial_counts))
    counts = burial_counts[tmpl_idx].astype(np.float64)     # enterrement des positions gabarit alignées
    kd_vals = KD_VEC[seq_idx]                               # hydropathie des AA alignés (déjà sous-sélectionnés)

    return float(w_burial * np.sum(kd_vals * (mu - counts)))


def score_threading_energy(aln_path: list[tuple[int | None, int | None]],
                           dope: np.ndarray,
                           D_tmpl: np.ndarray,
                           seq_idx: np.ndarray,
                           cutoff: float = PAIRWISE_CUTOFF,
                           burial_counts: np.ndarray | None = None,
                           w_burial: float = W_BURIAL) -> float:
    """
    Calculer l'énergie finale du threading :

        E_total =  E_pairwise_DOPE (pairs alignées ; d ≤ cutoff)
                 + E_burial * 1[w_burial≠0]

    Parameters
    ----------
    aln_path : list of tuple(int|None, int|None)
        Chemin global (NW/Gotoh).
    dope : numpy.ndarray, shape (20, 20, N_BINS)
        Tenseur DOPE (énergies ; plus petit = mieux).
    D_tmpl : numpy.ndarray, shape (M, M)
        Matrice des distances Cα-Cα du gabarit.
    seq_idx : numpy.ndarray of int, shape (N,)
        Séquence **complète** encodée (0..19).
    cutoff : float, optional
        Cutoff (Å) pour compter une paire.
    burial_counts : numpy.ndarray or None, shape (M,), optional
        Compte de voisins par résidu (proxy d'enterrement). Requis si `w_burial ≠ 0`.
    w_burial : float, optional
        Poids du terme d'enterrement (0.0 = off).

    Returns
    -------
    float
        Énergie totale (plus **petit** = meilleur).
    """
    tmpl_idx, seq_aln_idx = _alignment_matches_only(aln_path)
    if tmpl_idx.size == 0:
        return 0.0

    # (1) DOPE paire-à-paire limité aux positions alignées
    E_pair = energy_pairwise_dope(dope, D_tmpl, seq_idx, tmpl_idx, seq_aln_idx, cutoff=cutoff)

    # (2) Enterrement (optionnel)
    E_bur = 0.0
    if burial_counts is not None and w_burial != 0.0:
        # On passe uniquement les AA alignés (indices 0..19 dans l'ordre du chemin)
        seq_idx_aligned = seq_idx[seq_aln_idx]
        E_bur = energy_burial_term(seq_idx_aligned, tmpl_idx, burial_counts, w_burial=w_burial)

    return float(E_pair + E_bur)


def generate_decoy_energies(aln_path: list[tuple[int | None, int | None]],
                            dope: np.ndarray,
                            D_tmpl: np.ndarray,
                            seq_idx: np.ndarray,
                            n_decoys: int = N_DECOYS,
                            cutoff: float = PAIRWISE_CUTOFF,
                            rng: np.random.Generator | None = None) -> list[float]:
    """
    Générer des **décoys** en permutant aléatoirement les indices de séquence
    **uniquement parmi les positions alignées** (le set de positions gabarit reste fixe).

    Parameters
    ----------
    aln_path : list of tuple(int|None, int|None)
        Chemin global final.
    dope : numpy.ndarray, shape (20, 20, N_BINS)
        Tenseur DOPE.
    D_tmpl : numpy.ndarray, shape (M, M)
        Matrice des distances Cα-Cα du gabarit.
    seq_idx : numpy.ndarray of int, shape (N,)
        Séquence complète encodée (0..19).
    n_decoys : int, optional
        Nombre de décoys à générer.
    cutoff : float, optional
        Cutoff (Å) pour l'énergie DOPE paire-à-paire.
    rng : numpy.random.Generator or None, optional
        Générateur aléatoire (reproductibilité).

    Returns
    -------
    list[float]
        Liste des énergies DOPE des décoys.

    Notes
    -----
    - Ce schéma conserve la **densité d'appariements** et la **géométrie** du chemin,
      mais casse les associations AA↔positions → distribution nulle.
    - Sert à estimer la significativité via un **z-score**.
    """
    rng = rng or np.random.default_rng()
    tmpl_idx, seq_aln_idx = _alignment_matches_only(aln_path)
    if tmpl_idx.size == 0:
        return []

    L = tmpl_idx.size
    energies: list[float] = []
    for _ in range(n_decoys):
        perm = rng.permutation(L)
        seq_perm = seq_aln_idx[perm]
        E = energy_pairwise_dope(dope, D_tmpl, seq_idx, tmpl_idx, seq_perm, cutoff=cutoff)
        energies.append(float(E))
    return energies


def zscore_from_decoys(E_target: float, decoys: list[float]) -> float:
    """
    Calculer le **z-score** de l'énergie observée par rapport aux décoys :

        z = (E_target − mean(decoys)) / std(decoys)

    Interprétation (énergie plus basse = meilleure) :
      - z ≪ 0  → threading significatif (bien meilleur que le bruit)
      - z ≈ 0  → comparable aux décoys
      - z ≫ 0  → pire que la moyenne des décoys

    Parameters
    ----------
    E_target : float
        Énergie du threading.
    decoys : list[float]
        Énergies des décoys (≥ 2 pour avoir un écart-type).

    Returns
    -------
    float
        z-score (NaN si pas assez de décoys ou σ≈0).
    """
    if not decoys:
        return float("nan")
    mu = float(np.mean(decoys))
    sd = float(np.std(decoys, ddof=1)) if len(decoys) >= 2 else float("nan")
    if not np.isfinite(sd) or sd == 0.0:
        return float("nan")
    return (E_target - mu) / sd



# ==============================================
# 12 — Sauvegardes & graphiques
# ==============================================

def ensure_outdir(outdir: Path) -> Path:
    """
    Crée le répertoire de sortie s'il n'existe pas.

    Parameters
    ----------
    outdir : pathlib.Path

    Returns
    -------
    pathlib.Path
        Le même chemin, garanti existant.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def save_alignment_table(outpath: Path,
                         aln_path: list[tuple[int | None, int | None]],
                         res_ids: list[tuple[str, int, str]],
                         seq_str: str) -> None:
    """
    Écrire un tableau texte lisible de l'alignement global.

    Format
    ------
    Colonnes : step, tmpl_i, tmpl_res(3), seq_j, seq_aa, op
      - MATCH : (i, j)
      - GAP_V : (i, None)  # suppression côté structure
      - GAP_H : (None, j)  # insertion côté séquence
    """
    with outpath.open("w", encoding="utf-8") as f:
        f.write("# step\ttmpl_i\ttmpl_res(3)\tseq_j\tseq_aa\top\n")
        step = 0
        for i, j in aln_path:
            step += 1
            if i is not None and j is not None:
                res3, resseq, icode = res_ids[i]
                f.write(f"{step}\t{i}\t{res3}{resseq}{icode}\t{j}\t{seq_str[j]}\tMATCH\n")
            elif i is not None:
                res3, resseq, icode = res_ids[i]
                f.write(f"{step}\t{i}\t{res3}{resseq}{icode}\t-\t-\tGAP_V\n")
            else:
                f.write(f"{step}\t-\t-\t{j}\t{seq_str[j]}\tGAP_H\n")


def save_results_csv(outpath: Path,
                     template_name: str,
                     best_score_global: float,
                     energy_final: float,
                     zscore: float,
                     aln_path: list[tuple[int | None, int | None]],
                     M: int, N: int) -> None:
    """
    Écrire un CSV récapitulatif (scores + stats d'alignement).

    Columns
    -------
    template, score_global, E_final, zscore, aligned_len,
    gaps_vertical, gaps_horizontal, M_tmpl, N_seq
    """
    n_match = sum(1 for t in aln_path if t[0] is not None and t[1] is not None)
    n_gap_v = sum(1 for t in aln_path if t[0] is not None and t[1] is None)
    n_gap_h = sum(1 for t in aln_path if t[0] is None and t[1] is not None)
    header = ("template,score_global,E_final,zscore,aligned_len,"
              "gaps_vertical,gaps_horizontal,M_tmpl,N_seq\n")
    row = (f"{template_name},{best_score_global:.6f},{energy_final:.6f},{zscore:.6f},"
           f"{n_match},{n_gap_v},{n_gap_h},{M},{N}\n")
    with outpath.open("w", encoding="utf-8") as f:
        f.write(header)
        f.write(row)


def save_matrix_npy_and_png(mat: np.ndarray, basepath: Path, title: str | None = None, do_png: bool = True) -> None:
    """
    Sauver une matrice au format .npy et (optionnel) une heatmap .png.

    Parameters
    ----------
    mat : numpy.ndarray
        Matrice M×N à sauvegarder.
    basepath : pathlib.Path
        Chemin de base (sans extension).
    title : str or None, optional
        Titre de la figure.
    do_png : bool, optional
        Générer la heatmap PNG si True.
    """
    np.save(str(basepath) + ".npy", mat)
    if do_png:
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        im = ax.imshow(mat, aspect="auto")
        fig.colorbar(im, ax=ax)
        if title:
            ax.set_title(title)
        ax.set_xlabel("n (sequence)")
        ax.set_ylabel("m (template)")
        fig.tight_layout()
        fig.savefig(str(basepath) + ".png", dpi=150)
        plt.close(fig)



# =======================
# Helpers de tests
# =======================

def _check_dope_tensor(dope: np.ndarray) -> None:
    assert dope.shape == (20, 20, N_BINS), f"Shape DOPE inattendue: {dope.shape}"
    # Symétrie
    sym_err = np.max(np.abs(dope - np.transpose(dope, (1, 0, 2))))
    LOGGER.info("Écart de symétrie max DOPE: %.6g", sym_err)
    # Sanity sur bins
    aa_ala = AA1_TO_INDEX["A"]
    sample_bins = dope[aa_ala, aa_ala, :10]
    LOGGER.info("DOPE[ALA,ALA][:10] = %s", np.array2string(sample_bins, precision=2, separator=", "))


def _demo_sequence(seq_idx: np.ndarray) -> None:
    LOGGER.info("Séquence encodée: longueur=%d ; premiers indices=%s ...",
                len(seq_idx), seq_idx[:10])



# =======================
# MAIN
# =======================

def main():
    # -----------------------
    # 0) CLI & seed (repro)
    # -----------------------
    parser = build_parser()
    args = parser.parse_args()

    # Graîne RNG optionnelle (permet de reproduire les décoys, les sélections, etc.)
    if args.seed is not None:
        np.random.seed(args.seed)

    # -----------------------
    # 1) Charger DOPE (S1)
    # -----------------------
    # Si l’argument --dope n’est pas fourni, on tente quelques emplacements par défaut.
    dope_path = args.dope
    if dope_path is None:
        cand = [Path("/mnt/data/dope.par"), Path("./dope.par")]
        dope_path = next((p for p in cand if p.exists()), cand[0])
    LOGGER.info("Chargement DOPE depuis: %s", dope_path)

    # dope.par → tenseur DOPE[20,20,N_BINS] (Cα–Cα) + vérification de forme
    dope = load_dope_par(dope_path)
    _check_dope_tensor(dope)

    # ------------------------------------------
    # 2) Lire la séquence (& encoder) (S1–S2)
    # ------------------------------------------
    # Cas nominal : une séquence FASTA est fournie. Sinon, séquence jouet (pour démo).
    if args.fasta and args.fasta.exists():
        seq_str = read_fasta_one(args.fasta)
        LOGGER.info("Séquence FASTA lue (%d aa) depuis %s", len(seq_str), args.fasta)
    else:
        seq_str = "ACDEFGHIKLMNPQRSTVWY"  # jouet
        LOGGER.info("Aucun FASTA fourni: utilisation d'une séquence jouet (%d aa)", len(seq_str))

    # Encodage 1-lettre → indices 0..19 (ex. 'A'→0, 'R'→1, …)
    seq_idx = encode_sequence(seq_str)

    # Option pédagogique : petit aperçu (composition, indices…)
    _demo_sequence(seq_idx)
    LOGGER.info("Sprints 1-2 OK! — DOPE chargé & séquence encodée.")

    # ------------------------------------------------------
    # 3) Charger le gabarit PDB & distances Cα–Cα (S3–S4)
    # ------------------------------------------------------
    # Sans PDB on s’arrête proprement après les sprints 1–2.
    if args.pdb is None or not args.pdb.exists():
        LOGGER.warning("Aucun PDB fourni. Passe --pdb <file.pdb> [--chain A] pour continuer jusqu'aux sprints 3+.")
        LOGGER.info("Sprints 1-2 OK! — arrêt volontaire ici.")
        return

    LOGGER.info("Chargement du gabarit PDB: %s (chain=%s)", args.pdb, args.chain or "auto")
    coords, res_ids = load_template_ca(args.pdb, chain_id=args.chain)
    M = coords.shape[0]
    LOGGER.info("Cα extraits: M=%d ; premier résidu: %s", M, res_ids[0] if res_ids else "n/a")

    # Matrice des distances euclidiennes entre Cα du gabarit
    D = build_distance_matrix(coords)

    # Quelques stats rapides (symétrie, borne min/max, quartiles) pour sanity-check
    sym_err = float(np.max(np.abs(D - D.T)))
    offdiag = D[~np.eye(M, dtype=bool)]
    d_min = float(np.min(offdiag)) if offdiag.size else 0.0
    d_max = float(np.max(offdiag)) if offdiag.size else 0.0
    q25, q50, q75 = np.percentile(offdiag, [25, 50, 75]) if offdiag.size else (0, 0, 0)
    LOGGER.info("Matrice D: sym_err=%.3g ; d_min(offdiag)=%.2f Å ; d_max=%.2f Å ; quartiles=[%.2f, %.2f, %.2f] Å",
                sym_err, d_min, d_max, q25, q50, q75)

    # Distribution du “degré” Cα à 8Å/10Å → donne une idée de la densité locale (noyau vs surface)
    for r in (8.0, 10.0):
        neigh_counts = np.sum(D <= r, axis=1) - 1
        LOGGER.info("Rayon %.1f Å — voisins Cα: min=%d, median=%d, max=%d",
                    r, int(neigh_counts.min()), int(np.median(neigh_counts)), int(neigh_counts.max()))
    LOGGER.info("Sprints 3-4 OK! — Cα chargés et matrice de distances calculée.")

    # ----------------------------------------------------------------------
    # 4) DDP local (S5–S7) — test ponctuel pour valider L + Smith–Waterman
    # ----------------------------------------------------------------------
    # On teste une seule hypothèse (m↔n) au milieu, pour vérifier le pipeline local.
    m_test = min(M // 2, M - 1)
    n_test = min(len(seq_idx) // 2, len(seq_idx) - 1)
    LOGGER.info("Test local DDP: m=%d, n=%d, q_window=%d", m_test, n_test, DEFAULT_Q_WINDOW)

    # Sonde de binning (distance→bin DOPE)
    for d_probe in (0.4, 1.0, 3.9, 9.9, 10.1, 14.9):
        LOGGER.info("distance_to_bin(%.1f Å) -> %d", d_probe, distance_to_bin(d_probe))

    # L(m,n) : voisins de m vs fenêtre autour de n ; puis SW local (affine)
    L, P_list, Q_list = lowlevel_env_matrix(dope, D, seq_idx, m_test, n_test,
                                            q_window=DEFAULT_Q_WINDOW, cutoff=args.cutoff)
    LOGGER.info("L shape = %s ; |P|=%d ; |Q|=%d", L.shape, len(P_list), len(Q_list))
    if L.size:
        sw_score, sw_path = smith_waterman_local(L, gap_open=GAP_OPEN_LOCAL, gap_ext=GAP_EXT_LOCAL)
        LOGGER.info("SW local: score=%.3f ; path_len=%d", sw_score, len(sw_path))
        if len(sw_path) > 0:
            # On illustre 5 pas du chemin local avec la distance d(m,p) correspondante
            preview = []
            for (ip, iq) in sw_path[:5]:
                p_real = P_list[ip]; q_real = Q_list[iq]
                preview.append((p_real, q_real, float(D[m_test, p_real])))
            LOGGER.info("Exemples (p_real, q_real, d_mp[Å]) sur le chemin: %s", preview)
    LOGGER.info("Sprints 5-7 OK! — DOPE lookup, matrice L, SW local.")

    # ----------------------------------------------------------------
    # 5) Q initial & DDP global (S8–S9) — première passe de threading
    # ----------------------------------------------------------------
    # Graines (m,n) réparties le long de la diagonale proportionnelle
    selection = initial_pair_selection(seq_idx, D, sse=None, K0=K0_SELECTION)
    LOGGER.info("Sélection initiale: %d paires. Exemples: %s", len(selection), selection[:5])

    # Construction parcimonieuse de Q : Q(m,n) uniquement sur la sélection
    Q, qinfo = compute_Q_matrix(dope, D, seq_idx, selection,
                                q_window=DEFAULT_Q_WINDOW, cutoff=args.cutoff)
    LOGGER.info("Q: n_eval=%d ; mean=%.3f ; top5=%s", qinfo["n_evaluated"], qinfo["mean_score"], qinfo["top5"])

    # Première matrice H (ici H = Q “nu”) suivie d’un NW/Gotoh global
    H = Q
    best_score, aln_path = nw_gotoh_global(H, sse=None)

    # Petites stats d’alignement (matches/gaps)
    n_match = sum(1 for t in aln_path if t[0] is not None and t[1] is not None)
    n_gap_v = sum(1 for t in aln_path if t[0] is not None and t[1] is None)
    n_gap_h = sum(1 for t in aln_path if t[0] is None and t[1] is not None)
    LOGGER.info("NW/Gotoh global: score=%.3f ; len(path)=%d ; matches=%d ; gapsV=%d ; gapsH=%d",
                best_score, len(aln_path), n_match, n_gap_v, n_gap_h)
    LOGGER.info("Aperçu chemin global (10 ops): %s", aln_path[:10])
    LOGGER.info("Sprints 8-9 OK! — alignement global.")

    # ------------------------------------------------------------
    # 6) Boucle itérative DDP (S10) — sélection ⇄ renforcement H
    # ------------------------------------------------------------
    # À chaque cycle :
    #  - Local : (Q [+R si SAP])
    #  - Global : NW/Gotoh sur H = Q + α·Bias (+ β·R)
    #  - Sélection : élargir autour du chemin + ajouter pics de H
    #  - Mémoire : Bias ← γ·Bias + (1−γ)·normalize(H)
    aln_path_it, best_score_it, H_final = iterate_ddp(
        dope, D, seq_idx, sse=None,
        K0=K0_SELECTION, cycles=args.cycles, expand_k=args.expand_k,
        q_window=args.q_window, cutoff=args.cutoff,
        alpha_bias=args.alpha_bias, gamma_bias=args.gamma_bias,
        use_sap=args.sap_reinforce, beta_sap=args.sap_beta, sap_norm=args.sap_norm,
    )

    # Stats finales de l’alignement itératif
    n_match_it = sum(1 for t in aln_path_it if t[0] is not None and t[1] is not None)
    n_gap_v_it = sum(1 for t in aln_path_it if t[0] is not None and t[1] is None)
    n_gap_h_it = sum(1 for t in aln_path_it if t[0] is None and t[1] is not None)
    LOGGER.info("[Itératif] score=%.3f ; len(path)=%d ; matches=%d ; gapsV=%d ; gapsH=%d",
                best_score_it, len(aln_path_it), n_match_it, n_gap_v_it, n_gap_h_it)
    LOGGER.info("[Itératif] chemin (10 ops): %s", aln_path_it[:10])
    LOGGER.info("Sprint 10 OK! — orchestration.")

    # -------------------------------------------------
    # 7) Énergie finale & z-score (S11) — évaluation
    # -------------------------------------------------
    # Proxy d’enterrement (nombre de voisins Cα par résidu gabarit)
    burial_counts = burial_proxy_counts(D, radius=args.burial_radius)

    # Énergie DOPE paire-à-paire (≤ cutoff) + (option) terme enterrement
    E_final = score_threading_energy(
        aln_path_it, dope, D, seq_idx,
        cutoff=args.cutoff, burial_counts=burial_counts, w_burial=args.w_burial
    )

    # Distribution nulle par décoys (permutation des AA alignés)
    decoys = generate_decoy_energies(
        aln_path_it, dope, D, seq_idx, n_decoys=args.n_decoys, cutoff=args.cutoff
    )

    # z-score : significativité de E_final par rapport aux décoys
    z = zscore_from_decoys(E_final, decoys)
    LOGGER.info("Énergie finale (≤ %.1f Å): E=%.3f ; z=%.3f (n_decoys=%d)",
                args.cutoff, E_final, z, len(decoys))
    LOGGER.info("Sprint 11 OK! — énergie + z-score.")

    # ------------------------------------------
    # 8) Sauvegardes (S12) — CSV / align / mats
    # ------------------------------------------
    outdir = ensure_outdir(args.outdir)

    # Résumé chiffré (scores + stats d’aln)
    save_results_csv(outdir / "results.csv", args.pdb.stem, best_score_it, E_final, z,
                     aln_path_it, M, len(seq_idx))

    # Alignement lisible (table) pour inspection manuelle
    save_alignment_table(outdir / "best_alignment.txt", aln_path_it, res_ids, seq_str)

    # (Option) Dump des matrices Q et H_final (npy + png) pour diagnostic
    if args.save_q:
        save_matrix_npy_and_png(Q, outdir / "Q", "Q(m,n)", do_png=args.plot)
    if args.save_h:
        save_matrix_npy_and_png(H_final, outdir / "H_final", "H_final(m,n)", do_png=args.plot)

    LOGGER.info("Résultats sauvegardés dans: %s", outdir)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Threading par double programmation dynamique (DDP) — exécution unitaire",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Fichiers d'entrée
    ap.add_argument("--dope", type=Path, default=None, help="Chemin vers dope.par")
    ap.add_argument("--fasta", type=Path, default=None, help="Séquence cible (FASTA)")
    ap.add_argument("--pdb",   type=Path, default=None, help="Gabarit PDB pour tests/sorties")
    ap.add_argument("--chain", type=str,  default=None, help="ID de chaîne du PDB (auto si unique)")

    # Sorties / options de sauvegarde
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Dossier de sortie")
    ap.add_argument("--save-q", action="store_true", help="Sauver Q en .npy (+ .png si --plot)")
    ap.add_argument("--save-h", action="store_true", help="Sauver H_final en .npy (+ .png si --plot)")
    ap.add_argument("--plot",   action="store_true", help="Générer des figures PNG")

    # Itération / renforcement
    ap.add_argument("--sap-reinforce", action="store_true", help="Activer le renforcement SAP-like")
    ap.add_argument("--sap-beta", type=float, default=SAP_BETA, help="Poids β de R dans H=Q+αBias+βR")
    ap.add_argument("--sap-norm", type=str, default=SAP_NORM, choices=["length","none"], help="Normalisation de R")

    # Hyperparams (laisser les valeurs actuelles par défaut)
    ap.add_argument("--cycles",   type=int,   default=N_CYCLES,       help="Nombre max de cycles")
    ap.add_argument("--expand-k", type=int,   default=EXPAND_K,       help="Pics H ajoutés par cycle")
    ap.add_argument("--q-window", type=int,   default=DEFAULT_Q_WINDOW, help="Demi-fenêtre locale SW")
    ap.add_argument("--cutoff",   type=float, default=PAIRWISE_CUTOFF, help="Cutoff des paires DOPE (Å)")
    ap.add_argument("--alpha-bias", type=float, default=ALPHA_BIAS,   help="Poids du biais accumulé")
    ap.add_argument("--gamma-bias", type=float, default=GAMMA_BIAS,   help="Facteur d'oubli du biais")

    # Énergie / décoys
    ap.add_argument("--n-decoys", type=int, default=N_DECOYS, help="Nombre de décoys pour le z-score")
    ap.add_argument("--burial-radius", type=float, default=BURIAL_RADIUS, help="Rayon pour proxy d'enterrement (Å)")
    ap.add_argument("--w-burial", type=float, default=W_BURIAL, help="Poids du terme d'enterrement (0=off)")

    # Divers
    ap.add_argument("--seed", type=int, default=None, help="Graine aléatoire (reproductibilité)")

    return ap



if __name__ == "__main__":
    main()
