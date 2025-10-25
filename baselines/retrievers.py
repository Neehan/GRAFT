"""Retriever implementations: GRAFT, Zero-shot Dense, BM25."""

import logging
from pathlib import Path
from typing import Literal, Union

import faiss

import numpy as np
import torch
import bm25s
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

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
    """Zero-shot retriever: pretrained encoder + FAISS index built on the fly."""

    def __init__(self, model_name, embeddings_source, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.num_gpus = torch.cuda.device_count()
        self.index_config = config["index"]
        self.search_batch_size = int(self.index_config["gpu_search_batch_size"])
        if self.search_batch_size <= 0:
            raise ValueError(
                f"index.gpu_search_batch_size must be positive: {self.search_batch_size}"
            )
        self._load_encoder(model_name)
        embeddings = self._load_embeddings(embeddings_source)
        embeddings = self.normalize_embeddings(embeddings)
        device_pref = "cuda" if self.device.type == "cuda" else "cpu"
        self.index = self.build_index(embeddings, self.index_config, device=device_pref)
        logger.info(f"FAISS query chunk size set to {self.search_batch_size}")

    def _load_encoder(self, model_name):
        logger.info(f"Loading zero-shot encoder: {model_name}")
        encoder = SentenceTransformer(model_name, device=str(self.device))
        encoder.max_seq_length = self.config["encoder"]["max_len"]
        encoder.eval()

        if self.num_gpus > 1:
            logger.info(f"Using {self.num_gpus} GPUs with DataParallel for encoder")
            self.encoder = torch.nn.DataParallel(encoder)
            self._base_encoder = encoder
        else:
            self.encoder = encoder
            self._base_encoder = encoder

    def _load_embeddings(self, source: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(source, np.ndarray):
            return source
        elif isinstance(source, str):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Embeddings not found: {path}")
            logger.info(f"Loading embeddings from {path}")
            return np.load(path)
        else:
            raise TypeError(
                f"Unsupported embeddings source type: {type(source)} (expected str or np.ndarray)"
            )

    @staticmethod
    def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """Convert embeddings to contiguous float32 and L2-normalize."""
        if embeddings.dtype != np.float32:
            logger.info("Converting embeddings to float32 for FAISS")
            embeddings = embeddings.astype(np.float32)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        return np.ascontiguousarray(embeddings, dtype=np.float32)

    @staticmethod
    def build_index(
        embeddings: np.ndarray,
        index_config: dict,
        *,
        device: Literal["cuda", "cpu"] = "cuda",
    ) -> faiss.Index:
        """Construct a sharded IndexFlatIP across all visible GPUs."""
        dim = embeddings.shape[1]
        base_index = faiss.IndexFlatIP(dim)

        use_gpu = bool(index_config["use_gpu"])
        use_fp16 = bool(index_config["use_fp16"])
        logger.info("FAISS index config: use_gpu=%s, use_fp16=%s", use_gpu, use_fp16)

        if device != "cuda" or not use_gpu:
            logger.info("Building CPU IndexFlatIP")
            base_index.add(embeddings)
            return base_index

        if not hasattr(faiss, "get_num_gpus") or not hasattr(
            faiss, "index_cpu_to_all_gpus"
        ):
            raise RuntimeError(
                "FAISS GPU support is required for IndexFlatIP but is not available in this build"
            )

        num_gpus = faiss.get_num_gpus()
        if num_gpus == 0:
            logger.info("No GPUs visible to FAISS; falling back to CPU index")
            base_index.add(embeddings)
            return base_index

        clone_opts = faiss.GpuMultipleClonerOptions()
        clone_opts.useFloat16 = use_fp16
        clone_opts.shard = True

        logger.info(
            f"Moving flat index to all {num_gpus} visible GPUs (fp16={use_fp16})"
        )
        gpu_index = faiss.index_cpu_to_all_gpus(base_index, clone_opts)

        logger.info(f"Adding {embeddings.shape[0]:,} vectors on GPU")
        gpu_index.add(embeddings)
        return gpu_index

    def search(self, queries, k):
        is_single = isinstance(queries, str)
        if is_single:
            queries = [queries]

        with torch.no_grad():
            query_embeds = self._base_encoder.encode(
                queries,
                convert_to_tensor=False,
                normalize_embeddings=self.config["encoder"]["normalize_embeddings"],
                batch_size=self.config["encoder"]["eval_batch_size"],
                show_progress_bar=False,
                device=str(self.device),
            )
            query_embeds = np.ascontiguousarray(query_embeds, dtype=np.float32)

            if query_embeds.shape[0] <= self.search_batch_size:
                _, indices = self.index.search(query_embeds, k)
            else:
                indices_batches = []
                for start in range(0, query_embeds.shape[0], self.search_batch_size):
                    end = min(start + self.search_batch_size, query_embeds.shape[0])
                    _, idx = self.index.search(query_embeds[start:end], k)
                    indices_batches.append(idx)
                indices = np.vstack(indices_batches)

        results = [idx.tolist() for idx in indices]
        return results[0] if is_single else results


class GRAFTRetriever(ZeroShotRetriever):
    """GRAFT retriever: trained encoder + FAISS index."""

    def _load_encoder(self, model_name):
        logger.info(f"Loading GRAFT encoder from {model_name}")
        encoder = SentenceTransformer(model_name, device=str(self.device))
        encoder.max_seq_length = self.config["encoder"]["max_len"]
        encoder.eval()

        if self.num_gpus > 1:
            logger.info(f"Using {self.num_gpus} GPUs with DataParallel for encoder")
            self.encoder = torch.nn.DataParallel(encoder)
            self._base_encoder = encoder
        else:
            self.encoder = encoder
            self._base_encoder = encoder


class BM25Retriever(BaseRetriever):
    """BM25 sparse retriever (bm25s backend)."""

    def __init__(self, corpus_texts):
        logger.info("Building BM25 index...")
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=True)
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens, show_progress=True)
        logger.info("BM25 index built")

    def search(self, queries, k):
        is_single = isinstance(queries, str)
        if is_single:
            queries = [queries]

        query_tokens = bm25s.tokenize(queries, stopwords="en")
        results, scores = self.retriever.retrieve(query_tokens, k=k)
        results = results.tolist()

        return results[0] if is_single else results
