# VeCo

Anonymous implementation of **To Veto or to Compensate:
Brand-Adaptive Neuro-Symbolic Composition for Commercial Site Selection**.

VeCo aligns frozen LLM-derived brand preferences with region evidence encoded
from an urban knowledge graph. It computes concept-level brand-region matches
and combines them with differentiable AND/OR operators.

## Repository layout

```text
.
|-- data/
|   |-- beijing/
|   `-- shanghai/
|-- pretrain_emb/
|-- vecorec/
|   |-- model.py
|   |-- layers.py
|   |-- load_data.py
|   |-- utils.py
|   |-- train.py
|   |-- gen_prior.py
|   |-- pretrain_rotate.py
|   |-- alpha_prior_beijing.npy
|   |-- alpha_prior_shanghai.npy
|   |-- region_kg_feats_beijing.npy
|   `-- region_kg_feats_shanghai.npy
`-- requirements.txt
```

The release includes the frozen brand priors used by the training pipeline.
Re-running the LLM is therefore not required.

## Environment

Python 3.9 or later and a CUDA-enabled PyTorch installation are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build appropriate for the local CUDA version before
installing `torch-geometric` when necessary.

## Training

Run all commands from `vecorec/`.

Beijing:

```bash
cd vecorec
python train.py \
  --dataset beijing \
  --seed 123 \
  --freeze_alpha true \
  --label_smoothing 0.8 \
  --filter_train \
  --tag beijing_reproduction
```

Shanghai:

```bash
cd vecorec
python train.py \
  --dataset shanghai \
  --seed 123 \
  --freeze_alpha true \
  --label_smoothing 0.9 \
  --filter_train \
  --tag shanghai_reproduction
```

Checkpoints and metric summaries are written to `vecorec/`. Generated files
are ignored by the supplied `.gitignore`.

## Optional prior regeneration

The bundled priors are sufficient for reproduction. To regenerate a prior
with a compatible DeepSeek endpoint:

```bash
python gen_prior.py \
  --dataset beijing \
  --llm_model deepseek-reasoner \
  --llm_key YOUR_API_KEY
```

The API key is passed only as a command-line argument and is not stored in the
repository. A local Ollama endpoint is also supported; see
`python gen_prior.py --help`.

## Data

Each city directory contains the UrbanKG triples, brand and region lists, and
the train/validation/test brand-region interactions used by the code.

## Anonymity

This artifact intentionally contains no author names, affiliations, machine
paths, API credentials, experiment logs, or repository history.
