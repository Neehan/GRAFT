# GRAFT
Graph Augmented Retriever Fine-Tuning

Fine-tune dual-encoder retrievers using graph message passing over document/entity graphs. Train with GNN to make embeddings relation-aware, then deploy encoder-only for standard ANN retrieval.

## Setup

```bash
conda env create -f environment.yml
conda activate graft
```

## Quickstart

GRAFT uses HuggingFace `datasets` to automatically download HotpotQA. No manual data download needed.

### 1. Prepare graph
```bash
bash scripts/prepare_data.sh ./data/hotpot
```

### 2. Train
```bash
bash scripts/run_train.sh configs/hotpot_e5_sage.yml
```

### 3. Embed corpus (encoder-only)
```bash
bash scripts/run_embed.sh \
  outputs/graft_hotpot_e5_sage/encoder_best.pt \
  configs/hotpot_e5_sage.yml \
  outputs/graft_hotpot_e5_sage/embeddings.npy
```

### 4. Build FAISS index
```bash
python -c "
import logging, yaml
logging.basicConfig(level=logging.INFO)
from graft.eval.build_faiss import build_faiss_index
with open('configs/hotpot_e5_sage.yml') as f: config = yaml.safe_load(f)
build_faiss_index('outputs/graft_hotpot_e5_sage/embeddings.npy', config, 'outputs/graft_hotpot_e5_sage/index.faiss')
"
```

### 5. Evaluate
```bash
bash scripts/run_eval.sh \
  outputs/graft_hotpot_e5_sage/encoder_best.pt \
  outputs/graft_hotpot_e5_sage/index.faiss \
  configs/hotpot_e5_sage.yml \
  outputs/graft_hotpot_e5_sage/results.json
```

## Structure

```
graft/
  data/          # Graph & query-doc pair builders
  models/        # Encoder (HF), GNN (GraphSAGE)
  train/         # Training loop, losses, sampler, hard-neg mining
  eval/          # Corpus embedding, FAISS, metrics (R@K, nDCG, MRR)
configs/         # YAML configs
scripts/         # Shell scripts for train/eval/prep
```

## Config

See [configs/hotpot_e5_sage.yml](configs/hotpot_e5_sage.yml) for full config schema (encoder, GNN, loss weights, training params).

## Losses

- **InfoNCE (q→d)**: Query-to-document supervised retrieval
- **Neighbor contrast**: Graph smoothness over edges
- **Link prediction** (optional): Edge prediction regularizer

Combined as: `L = λ L_q2d + (1-λ) L_nbr + α L_link`
