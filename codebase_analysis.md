# GRAFT Codebase Deep Analysis Report

**Date**: 2025-10-25
**Focus**: SentenceTransformer consistency, GPU utilization, multi-GPU training, bugs, and inefficiencies

---

## 1. SENTENCE TRANSFORMER USAGE - ✅ MOSTLY CONSISTENT

### Good
- All modules use `SentenceTransformer` consistently:
  - `graft/models/encoder.py:4`
  - `graft/train/train.py:23`
  - `baselines/retrievers.py:14`
  - `graft/data/augment_graph.py:9`
- `.encode()` API used throughout with proper parameters (normalize, batch_size, device)

### Issues

#### Redundant wrapper functions in `graft/models/encoder.py`
- `load_encoder()` and `get_embeddings()` are thin wrappers that add no value
- **Impact**: Code bloat, unused imports
- **Fix**: Remove entire file, use direct `SentenceTransformer` instantiation everywhere

#### Inconsistent device handling
- Some places: `str(device)`
- Others: direct device object
- **Fix**: Standardize to one approach (SentenceTransformer accepts both)

---

## 2. MULTI-GPU TRAINING - ⚠️ INCONSISTENT & INEFFICIENT

### Training (Accelerate DDP) - ✅ Good

**File**: `graft/train/train.py`

- Lines 54-59: Uses Accelerate properly for DDP across GPUs
- Lines 125-126: Distributed sampler shards data correctly
- Line 309: Unwraps model correctly for saving

```python
self.accelerator = Accelerator(
    gradient_accumulation_steps=self.config["train"]["gradient_accumulation_steps"],
    kwargs_handlers=[ddp_kwargs],
)
```

### Inference (DataParallel) - ❌ INCONSISTENT

**Problem**: Multi-GPU support is inconsistent across inference modules

#### ✅ Has multi-GPU support:
- `baselines/retrievers.py:61-67` (ZeroShotRetriever)
- `baselines/retrievers.py:176-182` (GRAFTRetriever)

```python
if self.num_gpus > 1:
    logger.info(f"Using {self.num_gpus} GPUs with DataParallel for encoder")
    self.encoder = torch.nn.DataParallel(encoder)
    self._is_parallel = True
```

#### ❌ Missing multi-GPU support:
- `graft/eval/embed_corpus.py:21` - Single device only
- `graft/data/augment_graph.py:28-34` - Single device only

**Impact**:
- Corpus embedding and graph augmentation are **4x slower** than they should be on multi-GPU setups
- Embedding 5.2M documents takes hours instead of minutes

---

## 3. GPU UTILIZATION - ⚠️ MAJOR INEFFICIENCIES

### Critical Issues

#### 1. `graft/eval/embed_corpus.py` - Single GPU only
**Location**: Lines 12-61

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = SentenceTransformer(encoder_path, device=str(device))  # Only uses 1 GPU
```

**Impact**:
- Embedding 5.2M documents on 1 GPU vs 4 GPUs = **4x slower**
- Blocks other GPU-accelerated operations

**Fix**: Add DataParallel wrapper like retrievers do

---

#### 2. `graft/data/augment_graph.py` - Single GPU only
**Location**: Lines 16-49

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = SentenceTransformer(model_name, device=str(device))  # Single GPU
```

**Impact**:
- kNN graph construction embeds all nodes on single GPU
- Bottleneck in data preparation pipeline

**Fix**: Parallelize embedding step with DataParallel

---

#### 3. `baselines/retrievers.py` - Redundant unwrapping
**Location**: Lines 146-147

```python
def search(self, queries, k):
    # ...
    encoder = self.encoder.module if self._is_parallel else self.encoder  # Every call!
    query_embeds = encoder.encode(...)
```

**Impact**:
- Repeated conditional check on every search call
- Code duplication (same pattern in GRAFTRetriever)

**Fix**: Unwrap once in `__init__` and store as `self._encoder`

---

#### 4. `graft/train/train.py` - Double unwrapping in dev eval
**Location**: Lines 263-269, 275-282

```python
def _evaluate(self):
    unwrapped_encoder = self.accelerator.unwrap_model(self.encoder)  # Unwrap #1

    corpus_embeds = unwrapped_encoder.encode(...)  # Line 263
    # ...
    query_embeds = unwrapped_encoder.encode(...)   # Line 275
```

**Impact**: Minor - already unwraps once, but could be cleaner

**Fix**: Single unwrap at top of function (already correct)

---

## 4. BUGS

### Critical

#### 1. `baselines/retrievers.py:44-47` - Batch size validation too strict

```python
self.search_batch_size = int(self.index_config["gpu_search_batch_size"])
if self.search_batch_size <= 0:
    raise ValueError(f"index.gpu_search_batch_size must be positive: {self.search_batch_size}")
```

**Problem**: Assumes `gpu_search_batch_size` always exists in config. Will crash with `KeyError` if missing.

**Fix**: Use `.get()` with default or validate config schema upfront

---

### Moderate

#### 2. `graft/train/train.py:156-164` vs `263-282` - Inconsistent encoding APIs

**Training loop** (lines 156-164):
```python
def _encode_texts(self, texts):
    embeddings = self.encoder.encode(...)  # Encodes through DDP wrapper
    return embeddings
```

**Dev eval** (lines 263-282):
```python
def _evaluate(self):
    unwrapped_encoder = self.accelerator.unwrap_model(self.encoder)
    corpus_embeds = unwrapped_encoder.encode(...)  # Unwraps first
```

**Problem**:
- Training encodes through DDP wrapper (adds gradient hooks)
- Dev eval unwraps first (more efficient)
- Inconsistent pattern

**Fix**: Always unwrap before calling `.encode()` to avoid DDP overhead

---

#### 3. `graft/data/augment_graph.py:143` - Inefficient degree computation

```python
old_degrees = torch.zeros(len(graph.node_text), dtype=torch.long)
if old_num_edges > 0:
    for node_id in graph.edge_index[0]:
        old_degrees[node_id] += 1
old_mean_degree = old_degrees.float().mean().item()
```

**Problem**: Python loop over millions of edges (O(E) with Python overhead)

**Fix**:
```python
old_degrees = torch.bincount(graph.edge_index[0], minlength=len(graph.node_text))
old_mean_degree = old_degrees.float().mean().item()
```

**Impact**: ~10-100x faster for large graphs

---

#### 4. `baselines/retrievers.py:156-161` - Unnecessary chunking for small queries

```python
indices_batches = []
for start in range(0, query_embeds.shape[0], self.search_batch_size):
    end = min(start + self.search_batch_size, query_embeds.shape[0])
    _, idx = self.index.search(query_embeds[start:end], k)
    indices_batches.append(idx)
indices = np.vstack(indices_batches)
```

**Problem**: When `len(queries) < search_batch_size`, creates single-iteration loop with `vstack` overhead

**Fix**: Skip chunking if `query_embeds.shape[0] <= self.search_batch_size`

---

### Minor

#### 5. `graft/models/encoder.py` - Unused file

**Problem**:
- Functions `load_encoder()` and `get_embeddings()` are never imported
- Violates DRY principle - everyone reimplements encoder loading

**Fix**: Remove file entirely

---

#### 6. `graft/train/train.py:101-104` - Device redundancy

```python
self.encoder = SentenceTransformer(
    self.config["encoder"]["model_name"],
    device=str(self.device)  # Device already set
)
```

Later:
```python
embeddings = self.encoder.encode(
    texts,
    device=str(self.device)  # Device passed again
)
```

**Problem**: SentenceTransformer already knows its device from `__init__`, doesn't need it in `.encode()`

**Fix**: Remove `device` parameter from `.encode()` calls

---

## 5. PERFORMANCE OPTIMIZATIONS

### High Impact

#### 1. Parallelize `graft/eval/embed_corpus.py`

**Current**:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = SentenceTransformer(encoder_path, device=str(device))
```

**Proposed**:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = SentenceTransformer(encoder_path, device=str(device))
num_gpus = torch.cuda.device_count()

if num_gpus > 1:
    logger.info(f"Using {num_gpus} GPUs with DataParallel")
    encoder = torch.nn.DataParallel(encoder)
```

**Estimated speedup**: 3-4x on 4 GPUs

---

#### 2. Parallelize `graft/data/augment_graph.py`

**Current**: Single GPU embedding (lines 28-48)

**Proposed**: Same DataParallel wrapper as above

**Estimated speedup**: 3-4x on 4 GPUs

---

#### 3. `graft/train/train.py:156-164` - Avoid encoding overhead in training

**Current**:
```python
def _encode_texts(self, texts):
    embeddings = self.encoder.encode(...)  # Encodes through DDP wrapper
    return embeddings
```

**Proposed**:
```python
def _encode_texts(self, texts):
    unwrapped = self.accelerator.unwrap_model(self.encoder)
    embeddings = unwrapped.encode(...)  # Bypass DDP overhead
    return embeddings
```

**Impact**: Removes gradient hook overhead during forward pass

---

### Medium Impact

#### 4. `baselines/retrievers.py` - Cache unwrapped encoder

**Current**:
```python
def search(self, queries, k):
    encoder = self.encoder.module if self._is_parallel else self.encoder
    query_embeds = encoder.encode(...)
```

**Proposed**:
```python
def __init__(self, ...):
    # ... existing code ...
    self._encoder = self.encoder.module if self._is_parallel else self.encoder

def search(self, queries, k):
    query_embeds = self._encoder.encode(...)
```

**Impact**: Eliminates repeated conditional checks

---

#### 5. `graft/data/augment_graph.py:90-96` - Vectorize edge construction

**Current**:
```python
edges = []
for src_node in tqdm(range(num_nodes), desc="Building edges"):
    neighbors = indices[src_node][1:]
    for dst_node in neighbors:
        edges.append((src_node, int(dst_node)))
        edges.append((int(dst_node), src_node))
return list(set(edges))
```

**Proposed**:
```python
# Vectorized construction
src = np.repeat(np.arange(num_nodes), k)
dst = indices[:, 1:].flatten()

# Bidirectional edges
edges_fwd = np.stack([src, dst], axis=1)
edges_bwd = np.stack([dst, src], axis=1)
edges = np.unique(np.vstack([edges_fwd, edges_bwd]), axis=0)

return [(int(s), int(d)) for s, d in edges]
```

**Estimated speedup**: 10-100x for large graphs

---

#### 6. `graft/train/train.py:387` - Unnecessary `del` statements

```python
del batch, step_output, loss, loss_q2d, loss_nbr
```

**Problem**: Python's garbage collector handles this automatically. Adds clutter without benefit.

**Fix**: Remove `del` statements

---

## 6. CODE CONSISTENCY ISSUES

### 1. Device string conversion inconsistency

**Mixed usage**:
- `graft/train/train.py:103`: `device=str(self.device)`
- `graft/data/augment_graph.py:28`: `device=device`

**Fix**: Pick one style (both work with SentenceTransformer)

**Recommendation**: Use `str(device)` consistently for clarity

---

### 2. Encoder loading patterns

**All files use direct instantiation**:
- `graft/train/train.py:101-105`
- `baselines/retrievers.py:56-59`
- `graft/data/augment_graph.py:31-34`

**Except**: `graft/models/encoder.py` has wrapper functions (unused)

**Fix**: Remove `encoder.py`, use direct instantiation everywhere

---

### 3. `normalize_embeddings` handling

**Config**: `encoder.normalize_embeddings: true` (line 17)

**All `.encode()` calls pass it**: ✅ Good

**Issue**: `graft/eval/embed_corpus.py:57` converts to float16 before normalization is guaranteed

```python
all_embeddings = encoder.encode(
    graph.node_text,
    normalize_embeddings=config["encoder"]["normalize_embeddings"],  # Returns normalized
    ...
)
all_embeddings = all_embeddings.astype(np.float16)  # Then converts
```

**Current code is correct**, but could be clearer with a comment

---

## 7. ARCHITECTURAL CONCERNS

### 1. Training loop encodes through DDP wrapper

**Location**: `graft/train/train.py:156`

**Issue**:
- Adds gradient hooks even though we don't need them during encoding
- Slight performance overhead

**Recommendation**: Unwrap before `.encode()` like dev eval does

---

### 2. FAISS index not persisted

**Location**: `baselines/retrievers.py:111`

**Issue**:
- Rebuilds index every evaluation run
- For 5.2M corpus, this is expensive (several minutes)

**Recommendation**:
- Add optional index caching to disk
- Cache key: hash of embeddings path + config

---

### 3. No gradient checkpointing

**Location**: `graft/train/train.py`

**Issue**:
- With large subgraphs (512 nodes), memory could be an issue
- No gradient checkpointing for large models

**Recommendation**:
- Consider adding for models larger than e5-base
- Low priority unless training with large models

---

## SUMMARY: ACTION ITEMS BY PRIORITY

### Critical (Do First)

1. **Add multi-GPU support to `graft/eval/embed_corpus.py`**
   - Files: `graft/eval/embed_corpus.py`
   - Lines: 12-61
   - Impact: 4x speedup on multi-GPU systems

2. **Add multi-GPU support to `graft/data/augment_graph.py`**
   - Files: `graft/data/augment_graph.py`
   - Lines: 16-49
   - Impact: 4x speedup on multi-GPU systems

3. **Unwrap encoder before `.encode()` in training loop**
   - Files: `graft/train/train.py`
   - Lines: 156-164
   - Impact: Removes DDP overhead during encoding

---

### High Priority

4. **Remove unused `graft/models/encoder.py`**
   - Files: `graft/models/encoder.py` (entire file)
   - Impact: Code cleanliness, removes confusion

5. **Fix degree computation in `graft/data/augment_graph.py`**
   - Files: `graft/data/augment_graph.py`
   - Lines: 138-143, 160-164
   - Impact: 10-100x speedup

6. **Cache unwrapped encoder in retrievers**
   - Files: `baselines/retrievers.py`
   - Lines: 146-147 (ZeroShotRetriever), duplicated in GRAFTRetriever
   - Impact: Cleaner code, minor performance gain

---

### Medium Priority

7. **Vectorize edge construction in `graft/data/augment_graph.py`**
   - Files: `graft/data/augment_graph.py`
   - Lines: 90-96
   - Impact: 10-100x speedup for large graphs

8. **Standardize device handling**
   - Files: All encoder instantiations
   - Impact: Code consistency

9. **Remove unnecessary `del` statements**
   - Files: `graft/train/train.py`
   - Lines: 387
   - Impact: Code cleanliness

10. **Fix search chunking logic for small query batches**
    - Files: `baselines/retrievers.py`
    - Lines: 156-161
    - Impact: Minor performance gain

---

### Low Priority

11. **Add FAISS index persistence**
    - Files: `baselines/retrievers.py`
    - Impact: Speeds up repeated evaluations

12. **Consider gradient checkpointing**
    - Files: `graft/train/train.py`
    - Impact: Memory efficiency for large models

13. **Remove `device` parameter from `.encode()` calls**
    - Files: All encoder usage
    - Impact: Code cleanliness (device already set in `__init__`)

---

## FILES REQUIRING CHANGES

### Critical Priority
- `graft/eval/embed_corpus.py` (multi-GPU)
- `graft/data/augment_graph.py` (multi-GPU)
- `graft/train/train.py` (unwrap encoder)

### High Priority
- `graft/models/encoder.py` (DELETE)
- `graft/data/augment_graph.py` (vectorize degree computation)
- `baselines/retrievers.py` (cache unwrapped encoder)

### Medium Priority
- `graft/data/augment_graph.py` (vectorize edge construction)
- `graft/train/train.py` (remove del statements)
- `baselines/retrievers.py` (fix chunking logic)

---

## CONCLUSION

The codebase uses SentenceTransformer consistently but has **critical multi-GPU inefficiencies** in inference pipelines. The main bottlenecks are:

1. **Corpus embedding** runs on single GPU (should be 4x faster)
2. **Graph augmentation** runs on single GPU (should be 4x faster)
3. **Edge construction** uses Python loops (should be 10-100x faster)

These three fixes alone would reduce data preparation time from hours to minutes and significantly improve iteration speed during development.

The training loop uses Accelerate DDP correctly, but the inference/evaluation pipeline does not leverage available GPUs consistently.
