"""Deterministic cached dev-set construction for fast evaluation."""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Any

import torch


def _build_cache_key(meta: Dict[str, Any]) -> str:
    payload = json.dumps(meta, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def _sample_confusers(
    dev_pairs: List[Dict[str, Any]],
    pos_nodes: set,
    num_confusers: int,
    generator: torch.Generator,
) -> List[int]:
    if num_confusers <= 0:
        return []

    confuser_pool = []
    for pair in dev_pairs:
        for nid in pair["pos_nodes"]:
            if nid not in pos_nodes:
                confuser_pool.append(nid)

    if not confuser_pool:
        return []

    num_confusers = min(num_confusers, len(confuser_pool))
    perm = torch.randperm(len(confuser_pool), generator=generator)[:num_confusers]
    return [confuser_pool[idx.item()] for idx in perm]


def _sample_random_negatives(
    num_nodes: int,
    num_needed: int,
    banned: set,
    generator: torch.Generator,
) -> List[int]:
    negatives: List[int] = []
    if num_needed <= 0:
        return negatives

    draw_size = max(num_needed * 2, 1024)
    while len(negatives) < num_needed:
        sample = torch.randint(0, num_nodes, (draw_size,), generator=generator)
        for idx in sample.tolist():
            if idx not in banned:
                banned.add(idx)
                negatives.append(idx)
                if len(negatives) == num_needed:
                    break
        if draw_size >= num_nodes or draw_size > 4 * num_needed:
            draw_size = min(max(draw_size // 2, num_needed), num_nodes)

        if len(banned) >= num_nodes:
            break

    return negatives


def build_or_load_dev_set(
    cfg: Dict[str, Any],
    graph,
    dev_pairs: List[Dict[str, Any]],
    graph_path: str,
    is_main_process: bool,
    logger,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Build (or load) a deterministic dev set with cached corpus indices."""
    eval_cfg = cfg["eval"]
    train_seed = cfg["train"]["seed"]
    eval_seed = eval_cfg.get("seed", train_seed)
    num_dev_queries = min(len(dev_pairs), eval_cfg["num_samples"])
    dev_corpus_size = eval_cfg["dev_corpus_size"]
    confuser_fraction = eval_cfg.get("confuser_fraction", 0.1)

    cache_meta = {
        "graph_path": str(graph_path),
        "num_dev_queries": num_dev_queries,
        "dev_corpus_size": dev_corpus_size,
        "confuser_fraction": confuser_fraction,
        "seed": eval_seed,
    }

    cache_dir = Path(cfg["experiment"]["output_dir"]) / "dev_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _build_cache_key(cache_meta)
    cache_path = cache_dir / f"devset_{cache_key}.pt"

    if cache_path.exists():
        cached = torch.load(cache_path, weights_only=False)
        if cached.get("meta") == cache_meta:
            if is_main_process:
                logger.info(
                    "Loaded cached dev set (%d queries, %d corpus) from %s",
                    len(cached["dev_set"]),
                    len(cached["corpus_indices"]),
                    cache_path,
                )
            return cached["dev_set"], cached["corpus_indices"]

    if is_main_process:
        logger.info(
            "Building dev set (%d queries, corpus=%d) with confuser_fraction=%s",
            num_dev_queries,
            dev_corpus_size,
            confuser_fraction,
        )

    generator = torch.Generator()
    generator.manual_seed(eval_seed)

    num_nodes = len(graph.node_text)

    pos_nodes = set()
    for pair in dev_pairs[:num_dev_queries]:
        pos_nodes.update(pair["pos_nodes"])

    num_negatives = max(dev_corpus_size - len(pos_nodes), 0)
    max_available = max(num_nodes - len(pos_nodes), 0)
    if num_negatives > max_available and is_main_process:
        logger.warning(
            "Requested %d negatives but only %d unique negatives available; using all negatives.",
            num_negatives,
            max_available,
        )
        num_negatives = max_available

    num_confusers = int(num_negatives * confuser_fraction)
    confuser_pairs = dev_pairs[num_dev_queries:]
    confuser_nodes = _sample_confusers(
        confuser_pairs, pos_nodes, num_confusers, generator
    )

    used_nodes = set(pos_nodes)
    used_nodes.update(confuser_nodes)

    remaining_negatives = num_negatives - len(confuser_nodes)
    random_negatives = _sample_random_negatives(
        num_nodes, remaining_negatives, used_nodes, generator
    )

    neg_indices = confuser_nodes + random_negatives
    if len(neg_indices) < num_negatives and is_main_process:
        logger.warning(
            "Dev corpus negatives truncated: filled %d of %d requested.",
            len(neg_indices),
            num_negatives,
        )

    corpus_indices = list(pos_nodes) + neg_indices
    perm = torch.randperm(len(corpus_indices), generator=generator)
    corpus_indices = [corpus_indices[i.item()] for i in perm]

    dev_set: List[Dict[str, Any]] = []
    for pair in dev_pairs[:num_dev_queries]:
        gold_ids = set(pair["pos_nodes"])
        gold_positions = [
            idx for idx, corpus_id in enumerate(corpus_indices) if corpus_id in gold_ids
        ]
        dev_set.append({"query": pair["query"], "gold_positions": gold_positions})

    payload = {"meta": cache_meta, "dev_set": dev_set, "corpus_indices": corpus_indices}
    torch.save(payload, cache_path)

    if is_main_process:
        logger.info(
            "Dev set built and cached (%d queries, corpus=%d) at %s",
            len(dev_set),
            len(corpus_indices),
            cache_path,
        )

    return dev_set, corpus_indices
