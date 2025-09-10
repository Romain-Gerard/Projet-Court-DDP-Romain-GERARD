# DDP Threading

Implémentation du **threading par double programmation dynamique** :

* **Local** : Smith–Waterman sur matrices d’environnement (Cα–Cα + DOPE).
* **Global** : Needleman–Wunsch/Gotoh sur `H(m,n)`.
* Option **SAP-like** (renforcement le long des chemins locaux).
* Évaluation par **énergie DOPE** et **z-score** via décoys.

## Données requises

* `dope.par` : potentiels DOPE **Cα–Cα** (30 bins).
* `sequences/` : fichiers FASTA (`.fasta`) **non commités** (voir `.gitignore`).
* `templates/` : fichiers PDB (`.pdb`) **non commités**.

> Placez simplement les fichiers dans ces dossiers (ou passez des chemins absolus).

## Installation (uv)

```bash
# installer uv (Linux/macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc
echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
source ~/.bashrc

uv --version
uv self update

# installer les dépendances du projet (pyproject/uv.lock)
uv sync
```

## Utilisation

### 1) Processus unitaire (une séquence + un gabarit)

```bash
uv run python main.py \
  --dope dope.par \
  --fasta sequences/1BTA.fasta \
  --pdb templates/1BTA.pdb \
  --outdir ddp_out \
  --sap-reinforce --sap-beta 0.5 --sap-norm length --w-burial 0.3 \
  --n-decoys 100 --cycles 20 --plot --save-q --save-h --seed 123
```

**Sorties (dans `ddp_out/`)**

* `results.csv` : score global, énergie finale, z-score, stats d’alignement.
* `best_alignment.txt` : tableau lisible du chemin global.
* `Q.npy/.png` (option `--save-q`) : matrice `Q(m,n)`.
* `H_final.npy/.png` (option `--save-h`) : matrice finale utilisée par le global.

### 2) Évaluation (1 séquence vs ≥1 gabarits)

Prépare un **manifest** CSV (ex. `manif.csv`) :

```csv
pdb_path,chain,label,name,native
templates/1UBQ.pdb,A,1,1UBQ_A,1
templates/2XYZ.pdb,A,0,2XYZ_A,0
```

* `label` : 1 = gabarit positif (même pli/super-famille), 0 = négatif.
* `native` : 1 pour le gabarit “vrai” (optionnel, sert à reporter son rang).

Commande :

```bash
uv run python evaluate_threading.py \
  --manifest manif.csv \
  --fasta sequences/1UBQ.fasta \
  --dope dope.par \
  --outdir eval_out \
  --n-decoys 50 --sap-reinforce -j 8
```

**Sorties (dans `eval_out/`)**

* `per_template_results.csv` : par gabarit (score global, E, z, runtime…).
* `summary.csv` : métriques agrégées (Top-1/Top-5, ROC-AUC, PR-AUC, rang natif…).
* `roc.png`, `pr.png` : courbes ROC/PR.

## Paramètres utiles (rappel)

* `--cycles` : nb max de cycles itératifs (sélection ⇄ renforcement).
* `--expand-k` : nb de nouveaux points (pics de H + voisinage du chemin) ajoutés par cycle.
* `--sap-reinforce` / `--sap-beta` / `--sap-norm` : activer et régler le renforcement SAP-like.
* `--w-burial` : poids du petit terme “enterrement/solvation” (0 = désactivé).
* `--n-decoys` : nombre de décoys pour le z-score.
* `--plot`, `--save-q`, `--save-h` : figures/exports diagnostiques.
* `--seed` : reproductibilité (sélections, décoys).

## Dépannage rapide

* **`dope.par` introuvable** : placez-le à la racine du projet ou passez `--dope /chemin/vers/dope.par`.
* **PDB multi-chaînes** : passez `--chain A` (ou l’ID voulu).
* **Images vides** : vérifiez qu’il y a suffisamment de paires évaluées (`--expand-k`, `--cycles`) et que les chemins vers FASTA/PDB sont corrects.



