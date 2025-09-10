# DDP Threading (uv)

Implémentation du threading par **double DP** (SW local sur environnements + NW global), scoring **DOPE**, renforcement **SAP-like**, z-score via décoys.

## Installation (uv)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc
echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
source ~/.bashrc

uv --version
uv self update

uv add numpy pandas biopython matplotlib
uv sync
```

## Utilisation

Processus unitaire.

```bash
uv run python main.py \
  --dope dope.par \
  --fasta sequences/1BTA.fasta \
  --pdb templates/1BTA.pdb \
  --outdir ddp_out \
  --sap-reinforce --sap-beta 0.5 --sap-norm length --w-burial 0.3 \
  --n-decoys 100 --cycles 20 --plot --save-q --save-h --seed 123
```

Evaluation 1 séquence vs >=1 templates.

```bash
uv run python evaluate_threading.py \
  --manifest manif.csv \
  --fasta sequences/1UBQ.fasta \
  --dope dope.par \
  --outdir eval_out \
  --n-decoys 50 --sap-reinforce -j 8
```
