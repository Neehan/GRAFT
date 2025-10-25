"""Helpers for assembling dev sets with light confuser sampling."""

from typing import Dict, Iterable, List, Set, Tuple

import torch


def _collect_positive_nodes(dev_pairs: List[Dict], limit: int) -> Set[int]:
    pos_nodes: Set[int] = set()
    for pair in dev_pairs[:limit]:
        pos_nodes.update(pair["pos_nodes"])
    return pos_nodes


def _gather_confuser_candidates(
    dev_pairs: List[Dict],
    offset: int,
    positives: Set[int],
) -> List[int]:
    candidates: List[int] = []
    for pair in dev_pairs[offset:]:
        candidates.extend(nid for nid in pair["pos_nodes"] if nid not in positives)
    return candidates


def _sample_confusers(
    candidates: List[int],
    max_count: int,
    generator: torch.Generator,
) -> List[int]:
    if max_count <= 0 or not candidates:
        return []
    count = min(max_count, len(candidates))
    perm = torch.randperm(len(candidates), generator=generator)[:count]
    return [candidates[int(idx)] for idx in perm]


def _sample_random_negatives(
    num_nodes: int,
    needed: int,
    banned: Set[int],
    generator: torch.Generator,
) -> List[int]:
    negatives: List[int] = []
    if needed <= 0:
        return negatives

    draw_size = max(needed * 2, 1024)
    while needed > 0 and len(banned) < num_nodes:
        samples = torch.randint(0, num_nodes, (max(draw_size, 1),), generator=generator)
        for idx in samples.tolist():
            if idx not in banned:
                banned.add(idx)
                negatives.append(idx)
                needed -= 1
                if needed == 0:
                    break
        draw_size = max(min(draw_size // 2, num_nodes), 1)
    return negatives


def _build_dev_queries(
    queries: Iterable[Dict],
    corpus_indices: List[int],
) -> List[Dict[str, List[int]]]:
    dev_set: List[Dict[str, List[int]]] = []
    for pair in queries:
        gold_ids = set(pair["pos_nodes"])
        gold_positions = [
            idx for idx, corpus_id in enumerate(corpus_indices) if corpus_id in gold_ids
        ]
        dev_set.append({"query": pair["query"], "gold_positions": gold_positions})
    return dev_set


def build_dev_set(
    graph,
    dev_pairs: List[Dict],
    num_dev_queries: int,
    dev_corpus_size: int,
    confuser_fraction: float,
    seed: int,
    is_main_process: bool,
    logger,
) -> Tuple[List[Dict[str, List[int]]], List[int]]:
    """Create a deterministic dev corpus with cross-query confusers."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    positives = _collect_positive_nodes(dev_pairs, num_dev_queries)
    num_nodes = len(graph.node_text)

    num_negatives = max(dev_corpus_size - len(positives), 0)
    max_available = max(num_nodes - len(positives), 0)
    if num_negatives > max_available and is_main_process:
        logger.warning(
            "Requested %d negatives but only %d unique negatives available; using all negatives.",
            num_negatives,
            max_available,
        )
        num_negatives = max_available

    confuser_candidates = _gather_confuser_candidates(
        dev_pairs, num_dev_queries, positives
    )
    confuser_goal = int(num_negatives * confuser_fraction)
    confuser_nodes = _sample_confusers(confuser_candidates, confuser_goal, generator)

    used_nodes = set(positives)
    used_nodes.update(confuser_nodes)

    random_negatives = _sample_random_negatives(
        num_nodes=num_nodes,
        needed=num_negatives - len(confuser_nodes),
        banned=used_nodes,
        generator=generator,
    )

    if len(confuser_nodes) + len(random_negatives) < num_negatives and is_main_process:
        logger.warning(
            "Dev corpus negatives truncated: filled %d of %d requested.",
            len(confuser_nodes) + len(random_negatives),
            num_negatives,
        )

    corpus_indices = list(positives) + confuser_nodes + random_negatives
    perm = torch.randperm(len(corpus_indices), generator=generator)
    corpus_indices = [corpus_indices[i] for i in perm]

    dev_set = _build_dev_queries(dev_pairs[:num_dev_queries], corpus_indices)

    if is_main_process:
        logger.info(
            "Dev set built: %d queries, %d corpus nodes (%d positives)",
            len(dev_set),
            len(corpus_indices),
            len(positives),
        )

    return dev_set, corpus_indices
