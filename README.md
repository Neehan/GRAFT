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
pip install -e .
```

## Quickstart

GRAFT uses HuggingFace `datasets` to automatically download HotpotQA. No manual data download needed.

### 1. Prepare data
```bash
bash scripts/prepare_data.sh configs/hotpot_e5_sage.yml
```

### 2. Train
```bash
bash scripts/run_train.sh configs/hotpot_e5_sage.yml
```

If `semantic_k` is set and augmented graph doesn't exist, training will build it automatically. Augmented graphs are cached for reuse.

### 3. Evaluate GRAFT (embed + build index + compute metrics)
```bash
scripts/run_eval_graft.sh \
  outputs/graft_hotpot_e5_sage/encoder_best.pt \
  configs/hotpot_e5_sage.yml \
  outputs/graft_hotpot_e5_sage \
  validation
```

This runs the full GRAFT eval pipeline: embeds corpus with trained encoder, builds FAISS index, computes Recall@K/MRR metrics. Use `validation` split during development.

### 4. Run baselines (for final paper results)
```bash
scripts/run_baselines.sh configs/hotpot_e5_sage.yml outputs/baselines_test test
```

This runs all baselines on the test split (BM25, Zero-shot E5) and saves results. **Run this ONCE** for final paper evaluation.

## Structure

```
graft/
  data/          # Graph builders, kNN augmentation, query-doc pairs
  models/        # Encoder (HF), GNN (GraphSAGE)
  train/         # Training loop, losses, sampler
  eval/          # Corpus embedding, FAISS, metrics (R@K, nDCG, MRR)
baselines/       # BM25, Zero-shot retrievers
configs/         # YAML configs
scripts/         # Shell scripts for train/eval/prep/augment
```

## Config

See [configs/hotpot_e5_sage.yml](configs/hotpot_e5_sage.yml) for full config schema (encoder, GNN, loss weights, training params).

## Losses

- **InfoNCE (q→d)**: Query-to-document supervised retrieval
- **Neighbor contrast**: Graph smoothness over edges
- **Link prediction** (optional): Edge prediction regularizer

Combined as: `L = λ L_q2d + (1-λ) L_nbr + α L_link`
