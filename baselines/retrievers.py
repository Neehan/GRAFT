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

    def search(self, query, k):
        """Search for top-k documents given a query.

        Args:
            query: Query text string
            k: Number of documents to retrieve

        Returns:
            List of document IDs (indices into corpus)
        """
        raise NotImplementedError


class ZeroShotRetriever(BaseRetriever):
    """Zero-shot retriever: pretrained encoder + FAISS index."""

    def __init__(self, model_name, index_path, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self._load_encoder(model_name)
        self._load_faiss_index(index_path)

    def _load_encoder(self, model_name):
        logger.info(f"Loading zero-shot encoder: {model_name}")
        self.encoder = load_zero_shot_encoder(model_name, self.config, self.device)

    def _load_faiss_index(self, index_path):
        logger.info(f"Loading FAISS index from {index_path}")
        index = faiss.read_index(index_path)

        if self.device.type == "cuda":
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, index)
            logger.info("FAISS index on GPU")
        else:
            self.index = index

    def search(self, query, k):
        with torch.no_grad():
            query_embed = self.encoder.encode([query], self.device).cpu().numpy()
            query_embed = query_embed / np.linalg.norm(
                query_embed, axis=1, keepdims=True
            )
            distances, indices = self.index.search(query_embed, k)
            return indices[0].tolist()


class GRAFTRetriever(ZeroShotRetriever):
    """GRAFT retriever: trained encoder + FAISS index."""

    def _load_encoder(self, encoder_path):
        logger.info(f"Loading GRAFT encoder from {encoder_path}")
        self.encoder = load_trained_encoder(encoder_path, self.config, self.device)


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

    def search(self, query, k):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        return scores.argsort()[-k:][::-1].tolist()
