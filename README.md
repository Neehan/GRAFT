# GRAFT
Graph Augmented Retriever Fine-Tuning

Fine-tune dual-encoder retrievers using graph message passing over document/entity graphs. Train with GNN to make embeddings relation-aware, then deploy encoder-only for standard ANN retrieval.

## Setup

**Option 1: Local (with conda)**
```bash
conda env create -f environment.yml
conda activate graft
pip install -e .
```

**Option 2: SLURM (pip only)**
```bash
module load python/3.10  # or your cluster's Python module
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install faiss-gpu
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install -e .
```

## Quickstart

GRAFT uses HuggingFace `datasets` to automatically download HotpotQA. No manual data download needed.

### 1. Prepare data
```bash
bash scripts/prepare_data.sh ./datasets/hotpot
```

### 2. Train
```bash
bash scripts/run_train.sh configs/hotpot_e5_sage.yml
```

### 3. Evaluate (embed + build index + compute metrics)
```bash
bash scripts/run_eval.sh \
  outputs/graft_hotpot_e5_sage/encoder_best.pt \
  configs/hotpot_e5_sage.yml \
  outputs/graft_hotpot_e5_sage
```

This runs the full eval pipeline: embeds corpus, builds FAISS index, computes Recall@K/MRR metrics.

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
