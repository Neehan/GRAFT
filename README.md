# GRAFT
Graph Augmented Retriever Fine-Tuning

Fine-tune dual-encoder retrievers using graph structure over document graphs. Graph-aware contrastive learning makes embeddings relation-aware, then deploy encoder-only for standard ANN retrieval.

## Setup

**Option 1: Local (with conda)**
```bash
conda env create -f environment.yml
conda activate graft
pip install -e .
```

**Option 2: SLURM/Cluster (with existing conda)**
```bash
module load python/3.10  # or your cluster's Python module
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install faiss-gpu
pip install -e .

# If you get GLIBCXX errors, ensure conda's libstdc++ is used:
# All scripts already include: export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

## Quickstart

GRAFT uses HuggingFace `datasets` to automatically download `mteb/hotpotqa` (~5.2M documents). The full corpus is used for retrieval evaluation. Graph edges are built from document co-occurrence in the train split.

### 1. Prepare data
```bash
bash scripts/prepare_data.sh configs/hotpot_e5_sage.yml
```

This loads `mteb/hotpotqa`, filters the corpus to train split documents only, builds the base graph from document co-occurrence in queries, and augments it with kNN edges if `semantic_k` is set in the config. Prepared graphs are cached for reuse.

### 2. Train
```bash
bash scripts/run_train.sh configs/hotpot_e5_sage.yml
```

### 3. Evaluate GRAFT
```bash
bash scripts/run_eval_graft.sh \
  outputs/graft_hotpot_e5_sage/encoder_best.pt \
  configs/hotpot_e5_sage.yml \
  outputs/graft_hotpot_e5_sage \
  dev
```

Encodes corpus, builds FAISS, computes metrics. Set `eval.corpus_size: 500000` in config to sample instead of full 5.2M corpus (7min vs 1hr).

### 4. Run baselines
```bash
bash scripts/run_baselines.sh configs/hotpot_e5_sage.yml outputs/baselines_test test
```

Runs BM25 and Zero-shot E5. Zero-shot embeddings cached. Respects `eval.corpus_size`.

## Structure

```
graft/
  data/          # Graph builders, kNN augmentation, query-doc pairs
  models/        # Encoder (HF)
  train/         # Training loop, losses, sampler
  eval/          # Corpus embedding, FAISS, metrics (R@K, nDCG, MRR)
baselines/       # BM25, Zero-shot retrievers
configs/         # YAML configs
scripts/         # Shell scripts for train/eval/prep/augment
```

## Config

See [configs/hotpot_e5_sage.yml](configs/hotpot_e5_sage.yml) for full config schema (encoder, graph, loss weights, training params).

### FAISS Index

`IndexFlatIP` built on GPU, copied to CPU for search. In-memory, not persisted.

## Losses

- **InfoNCE (q→d)**: Query-to-document retrieval with hard negative mining (subgraph-level, per-batch)
- **Neighbor contrast**: Graph smoothness over edges
- **Link prediction** (optional): Edge prediction regularizer

### Training stability notes

- We log per-batch InfoNCE stats; if a query sees zero negatives the warning points at sampler issues.
- The sampler drops an extra random node into every subgraph so InfoNCE cannot degenerate into an all-positive softmax.
- Hard negatives are filtered to stay disjoint from the positives, and `loss.tau` is set to 0.1 in the default config to keep logits numerically stable.

Combined as: `L = λ L_q2d + (1-λ) L_nbr + α L_link`

Hard negatives mined from current batch subgraph via similarity ranking (no FAISS overhead).
