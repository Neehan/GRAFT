"""Retriever implementations: GRAFT, Zero-shot Dense, BM25."""

import logging
import torch
import numpy as np
import faiss
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from graft.models.encoder import load_trained_encoder, load_zero_shot_encoder

logger = logging.getLogger(__name__)


class BaseRetriever:
    """Base class for all retriever methods."""

    def search(self, queries, k):
        """Search for top-k documents given query(ies).

        Args:
            queries: Query text string or list of query strings
            k: Number of documents to retrieve

        Returns:
            List of document IDs, or list of lists for batch
        """
        raise NotImplementedError


class ZeroShotRetriever(BaseRetriever):
    """Zero-shot retriever: pretrained encoder + FAISS index."""

    def __init__(self, model_name, index_path, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.num_gpus = torch.cuda.device_count()
        self._load_encoder(model_name)
        self._load_faiss_index(index_path)

    def _load_encoder(self, model_name):
        logger.info(f"Loading zero-shot encoder: {model_name}")
        encoder = load_zero_shot_encoder(model_name, self.config, self.device)

        if self.num_gpus > 1:
            logger.info(f"Using {self.num_gpus} GPUs with DataParallel for encoder")
            self.encoder = torch.nn.DataParallel(encoder)
            self._is_parallel = True
        else:
            self.encoder = encoder
            self._is_parallel = False

    def _load_faiss_index(self, index_path):
        logger.info(f"Loading FAISS index from {index_path}")
        index = faiss.read_index(index_path)

        if self.device.type == "cuda":
            if self.num_gpus > 1:
                logger.info(f"Sharding FAISS index across {self.num_gpus} GPUs")
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                co.useFloat16 = True
                self.index = faiss.index_cpu_to_all_gpus(index, co=co)
                logger.info(
                    f"FAISS index sharded across {self.num_gpus} GPUs with FP16"
                )
            else:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("FAISS index on GPU 0")
        else:
            self.index = index

    def search(self, queries, k):
        is_single = isinstance(queries, str)
        if is_single:
            queries = [queries]

        with torch.no_grad():
            encoder = self.encoder.module if self._is_parallel else self.encoder
            query_embeds = encoder.encode(queries, self.device).cpu().numpy()
            query_embeds = query_embeds / np.linalg.norm(
                query_embeds, axis=1, keepdims=True
            )
            distances, indices = self.index.search(query_embeds, k)

        results = [idx.tolist() for idx in indices]
        return results[0] if is_single else results


class GRAFTRetriever(ZeroShotRetriever):
    """GRAFT retriever: trained encoder + FAISS index."""

    def _load_encoder(self, model_name):
        logger.info(f"Loading GRAFT encoder from {model_name}")
        encoder = load_trained_encoder(model_name, self.config, self.device)

        if self.num_gpus > 1:
            logger.info(f"Using {self.num_gpus} GPUs with DataParallel for encoder")
            self.encoder = torch.nn.DataParallel(encoder)
            self._is_parallel = True
        else:
            self.encoder = encoder
            self._is_parallel = False


class BM25Retriever(BaseRetriever):
    """BM25 sparse retriever."""

    def __init__(self, corpus_texts):
        logger.info("Building BM25 index...")

        def simple_tokenize(text):
            return text.lower().split()

        tokenized_corpus = [
            simple_tokenize(text) for text in tqdm(corpus_texts, desc="Tokenizing")
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, queries, k):
        is_single = isinstance(queries, str)
        if is_single:
            queries = [queries]

        results = []
        for query in queries:
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            results.append(scores.argsort()[-k:][::-1].tolist())

        return results[0] if is_single else results
