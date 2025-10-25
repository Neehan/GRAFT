"""Main training loop for GRAFT with Accelerate, wandb, and hard-negative mining."""

import logging
import warnings
import os

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import yaml
import torch
import wandb
from pathlib import Path
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from graft.models.encoder import Encoder
from graft.train.losses import compute_total_loss
from graft.train.sampler import GraphBatchSampler
from graft.train.hard_neg_miner import HardNegativeMiner
from graft.train.dev_utils import build_dev_set
from graft.data.pair_maker import load_query_pairs

logger = logging.getLogger("graft.train")


class GRAFTTrainer:
    def _log_memory(self, tag):
        """Log GPU memory usage."""
        if torch.cuda.is_available() and self.accelerator.is_main_process:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(
                f"[{tag}] GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
            )

    def __init__(self, config_path):
        self.config_path = config_path
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Set seed for reproducibility
        seed = self.config["train"]["seed"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config["train"][
                "gradient_accumulation_steps"
            ],
            kwargs_handlers=[ddp_kwargs],
        )
        self.device = self.accelerator.device

        if self.accelerator.is_main_process:
            # Build run name with kNN info
            base_name = self.config["experiment"]["name"]
            semantic_k = self.config["data"]["semantic_k"]
            knn_only = self.config["data"]["knn_only"]

            if semantic_k is not None:
                suffix = f"_knn_only{semantic_k}" if knn_only else f"_knn{semantic_k}"
                run_name = f"{base_name}{suffix}"
            else:
                run_name = base_name

            wandb.init(
                project="graft",
                name=run_name,
                config=self.config,
            )
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step")

        # self._log_memory("After accelerator init")
        self._setup_models()
        # self._log_memory("After setup_models")
        self._setup_data()
        # self._log_memory("After setup_data")
        self._setup_training()
        # self._log_memory("After setup_training")
        self._setup_hard_neg_miner()
        # self._log_memory("After setup_hard_neg_miner")

        self.global_step: int = 0
        self.best_recall: float = 0.0

    def _get_graph_path(self):
        """Get path to prepared graph."""
        graph_dir = Path(self.config["data"]["graph_dir"])
        graph_name = self.config["data"]["graph_name"]
        semantic_k = self.config["data"]["semantic_k"]
        knn_only = self.config["data"]["knn_only"]

        if semantic_k is None:
            graph_path = graph_dir / f"{graph_name}.pt"
        else:
            suffix = f"_knn_only{semantic_k}" if knn_only else f"_knn{semantic_k}"
            graph_path = graph_dir / f"{graph_name}{suffix}.pt"

        if not graph_path.exists():
            raise FileNotFoundError(
                f"Graph not found: {graph_path}\n"
                f"Run data preparation first: bash scripts/prepare_data.sh {self.config_path}"
            )

        if self.accelerator.is_main_process:
            logger.info(f"Using graph: {graph_path}")
        return str(graph_path)

    def _setup_models(self):
        self.encoder = Encoder(
            model_name=self.config["encoder"]["model_name"],
            max_len=self.config["encoder"]["max_len"],
            pool=self.config["encoder"]["pool"],
            freeze_layers=self.config["encoder"]["freeze_layers"],
        )

        self.tokenizer = self.encoder.tokenizer

    def _setup_data(self):
        if self.accelerator.is_main_process:
            logger.info("Loading mteb/hotpotqa datasets...")
        graph_path = self._get_graph_path()
        self.graph = torch.load(graph_path, weights_only=False)

        train_pairs = load_query_pairs(
            "train", graph_path, self.config, log=self.accelerator.is_main_process
        )

        with self.accelerator.main_process_first():
            self.sampler = GraphBatchSampler(
                graph=self.graph,
                train_pairs=train_pairs,
                query_batch_size=self.config["train"]["query_batch_size"],
                fanouts=self.config["graph"]["fanouts"],
                rank=self.accelerator.process_index,
                world_size=self.accelerator.num_processes,
            )

        if self.accelerator.is_main_process:
            logger.info(
                f"Distributed sampler: {self.accelerator.num_processes} processes, "
                f"{len(self.sampler)} batches per process, "
                f"{len(self.sampler) * self.accelerator.num_processes} total batches"
            )
        self.dev_data = self._load_fixed_dev_set()

    def _setup_training(self):
        optimizer = torch.optim.AdamW(  # type: ignore
            self.encoder.parameters(),
            lr=self.config["train"]["lr_encoder"],
            weight_decay=self.config["train"]["weight_decay"],
        )

        total_steps = len(self.sampler) * self.config["train"]["epochs"]
        warmup_steps = int(total_steps * self.config["train"]["warmup_ratio"])
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        self.encoder, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.encoder, optimizer, scheduler
        )

    def _setup_hard_neg_miner(self):
        """Initialize hard negative miner (subgraph-level)."""
        if self.config["train"]["hardneg_enabled"]:
            self.hard_neg_miner = HardNegativeMiner(self.config)
            if self.accelerator.is_main_process:
                logger.info("Hard negative miner enabled")
        else:
            self.hard_neg_miner = None

    def _encode_texts(self, texts):
        """Tokenize and encode texts in one forward pass."""
        encoded = self.tokenizer(
            texts,
            max_length=self.config["encoder"]["max_len"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(self.device)
        mask = encoded["attention_mask"].to(self.device)
        return self.encoder(ids, mask)

    def _load_fixed_dev_set(self):
        """Build small dev corpus for fast realistic evaluation."""
        graph_path = self._get_graph_path()
        dev_pairs = load_query_pairs("dev", graph_path, self.config, log=False)

        dev_config = self.config["dev"]
        dev_seed = dev_config["seed"]
        num_dev_queries = min(len(dev_pairs), dev_config["num_samples"])

        dev_set, corpus_indices = build_dev_set(
            graph=self.graph,
            dev_pairs=dev_pairs,
            num_dev_queries=num_dev_queries,
            dev_corpus_size=dev_config["dev_corpus_size"],
            confuser_fraction=dev_config["confuser_fraction"],
            seed=dev_seed,
            is_main_process=self.accelerator.is_main_process,
            logger=logger,
        )

        self.dev_corpus_indices = corpus_indices
        return dev_set

    def _training_step(self, batch):
        """Single training step."""
        # Encode all texts in single forward pass
        node_texts = [
            self.graph.node_text[int(nid)] for nid in batch["subgraph"].n_id_cpu.numpy()
        ]
        all_texts = batch["queries"] + node_texts
        all_embeds = self._encode_texts(all_texts)

        # Split embeddings (use split for proper gradient tracking)
        num_queries = len(batch["queries"])
        query_embeds, node_embeds = torch.split(
            all_embeds, [num_queries, len(node_texts)], dim=0
        )

        # Labels: Use all positives with soft-OR InfoNCE + masking (no repetition)
        subgraph_ids = batch["subgraph"].n_id.to(self.device)

        labels_list = []
        mask_list = []
        max_positives = max(len(pos_nodes) for pos_nodes in batch["pos_nodes"])

        for pos_nodes in batch["pos_nodes"]:
            pos_tensor = torch.tensor(pos_nodes, device=self.device, dtype=torch.long)
            matches = (subgraph_ids.unsqueeze(0) == pos_tensor.unsqueeze(1)).any(dim=0)
            pos_indices = matches.nonzero(as_tuple=False).squeeze(-1)

            if len(pos_indices) == 0:
                raise ValueError("No positives found for query in subgraph")

            # Pad with zeros and use mask (no repetition)
            padded = torch.zeros(max_positives, dtype=torch.long, device=self.device)
            mask = torch.zeros(max_positives, dtype=torch.bool, device=self.device)
            padded[: len(pos_indices)] = pos_indices
            mask[: len(pos_indices)] = True

            labels_list.append(padded)
            mask_list.append(mask)

        labels = torch.stack(labels_list, dim=0)
        labels_mask = torch.stack(mask_list, dim=0)

        # Mine hard negatives from subgraph
        hard_negs = None
        if self.hard_neg_miner is not None:
            # Convert pos_nodes (global IDs) to subgraph indices
            pos_indices_list = []
            for pos_nodes in batch["pos_nodes"]:
                pos_tensor = torch.tensor(
                    pos_nodes, device=self.device, dtype=torch.long
                )
                matches = (subgraph_ids.unsqueeze(0) == pos_tensor.unsqueeze(1)).any(
                    dim=0
                )
                pos_indices_list.append(
                    matches.nonzero(as_tuple=False).squeeze(-1).tolist()
                )

            hard_negs = self.hard_neg_miner.mine_hard_negatives(
                query_embeds, node_embeds, pos_indices_list
            )

        # Loss
        loss, loss_q2d, loss_nbr = compute_total_loss(
            query_embeds=query_embeds,
            doc_embeds=node_embeds,
            labels=labels,
            node_embeds=node_embeds,
            edge_index=batch["subgraph"].edge_index.to(self.device),
            pos_edges=batch["pos_edges"].to(self.device),
            neg_edges=batch["neg_edges"].to(self.device),
            labels_mask=labels_mask,
            hard_negs=hard_negs,
            **{
                k: self.config["loss"][k]
                for k in ["lambda_q2d", "tau", "tau_graph", "alpha_link"]
            },
        )

        return {"loss": loss, "loss_q2d": loss_q2d, "loss_nbr": loss_nbr}

    def _evaluate(self):
        """Fast realistic dev eval: retrieve from 100k corpus."""
        # self._log_memory("Start dev eval")
        self.encoder.eval()
        unwrapped_encoder = self.accelerator.unwrap_model(self.encoder)

        recall_k = self.config["dev"]["recall_k"]
        encoder_batch_size = self.config["encoder"]["dev_batch_size"]

        with torch.no_grad():
            # Encode dev corpus
            corpus_texts = [
                self.graph.node_text[idx] for idx in self.dev_corpus_indices
            ]
            corpus_embeds = []
            pbar = tqdm(
                total=len(corpus_texts),
                desc="Encoding dev corpus",
                disable=not self.accelerator.is_main_process,
            )
            # self._log_memory("Before corpus encoding")
            for i in range(0, len(corpus_texts), encoder_batch_size):
                batch = corpus_texts[i : i + encoder_batch_size]
                embeds = unwrapped_encoder.encode(batch, self.device)
                corpus_embeds.append(embeds.cpu())
                pbar.update(len(batch))
                # if i == 0:
                # self._log_memory("After first corpus batch")
            pbar.close()
            corpus_embeds = torch.cat(corpus_embeds, dim=0).to(self.device)  # (100k, D)
            # self._log_memory("After corpus encoding")

            # Encode queries in batches and compute top-k
            queries = [item["query"] for item in self.dev_data]
            all_top_k_indices = []

            pbar = tqdm(
                total=len(queries),
                desc="Encoding queries & searching",
                disable=not self.accelerator.is_main_process,
            )
            # self._log_memory("Before query encoding")
            for q_start in range(0, len(queries), encoder_batch_size):
                q_end = min(q_start + encoder_batch_size, len(queries))
                query_batch = queries[q_start:q_end]

                # Encode query batch
                query_embeds = unwrapped_encoder.encode(query_batch, self.device)

                # Compute scores for this query batch: (batch_size, corpus_size)
                scores = torch.matmul(query_embeds, corpus_embeds.T)

                # Get top-k per query
                _, top_k_indices = torch.topk(scores, k=recall_k, dim=1)
                all_top_k_indices.append(top_k_indices.cpu())
                pbar.update(len(query_batch))
                # if q_start == 0:
                # self._log_memory("After first query batch")
            pbar.close()

            all_top_k_indices = torch.cat(all_top_k_indices, dim=0)
            # self._log_memory("After query encoding")

            # Compute recall@k
            correct = 0
            for i, item in enumerate(self.dev_data):
                gold_set = set(item["gold_positions"])
                retrieved_set = set(all_top_k_indices[i].tolist())
                if gold_set & retrieved_set:
                    correct += 1

        return correct / len(self.dev_data)

    def _save_checkpoint(self, tag):
        """Save model checkpoints."""
        if self.accelerator.is_main_process:
            output_dir = Path(self.config["experiment"]["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            unwrapped_encoder = self.accelerator.unwrap_model(self.encoder)

            # Build filename with kNN info (same as wandb run name)
            semantic_k = self.config["data"]["semantic_k"]
            knn_only = self.config["data"]["knn_only"]

            if semantic_k is not None:
                suffix = f"_knn_only{semantic_k}" if knn_only else f"_knn{semantic_k}"
                filename = f"encoder_{tag}{suffix}.pt"
            else:
                filename = f"encoder_{tag}.pt"

            checkpoint_path = output_dir / filename
            torch.save(unwrapped_encoder.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    def train(self):
        """Main training loop."""
        if self.accelerator.is_main_process:
            logger.info("Running zero-shot evaluation...")
            zero_shot_recall = self._evaluate()
            logger.info(
                f"Zero-shot: dev_recall@{self.config['dev']['recall_k']}={zero_shot_recall:.4f}"
            )
            wandb.log({"global_step": 0, "dev_recall": zero_shot_recall})

        self.accelerator.wait_for_everyone()

        for epoch in range(self.config["train"]["epochs"]):
            self.encoder.train()

            total_steps = (
                len(self.sampler) // self.config["train"]["gradient_accumulation_steps"]
            )
            pbar = tqdm(
                total=total_steps,
                desc=f"Epoch {epoch+1}/{self.config['train']['epochs']}",
            )

            for batch in self.sampler:
                with self.accelerator.accumulate(self.encoder):
                    step_output = self._training_step(batch)
                    loss = step_output["loss"]
                    loss_q2d = step_output["loss_q2d"]
                    loss_nbr = step_output["loss_nbr"]

                    self.accelerator.backward(loss)

                    if self.config["train"]["grad_clip"] > 0:
                        self.accelerator.clip_grad_norm_(
                            self.encoder.parameters(),
                            self.config["train"]["grad_clip"],
                        )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    if self.accelerator.sync_gradients:
                        self.global_step += 1

                    if (
                        self.accelerator.sync_gradients
                        and self.global_step % self.config["train"]["log_every"] == 0
                    ):
                        if self.accelerator.is_main_process:
                            metrics = {
                                "global_step": self.global_step,
                                "loss": loss.item(),
                                "loss_q2d": loss_q2d.item(),
                                "loss_nbr": loss_nbr.item(),
                                "lr_encoder": self.scheduler.get_last_lr()[0],
                            }
                            wandb.log(metrics)
                            logger.info(
                                f"Step {self.global_step}: loss={loss.item():.4f}, loss_q2d={loss_q2d.item():.4f}, loss_nbr={loss_nbr.item():.4f}"
                            )

                    if self.accelerator.sync_gradients:
                        pbar.update(1)
                        pbar.set_postfix(
                            {"loss": f"{loss.item():.4f}", "step": self.global_step}
                        )

                    del batch, step_output, loss, loss_q2d, loss_nbr

                    if (
                        self.accelerator.sync_gradients
                        and self.global_step % self.config["train"]["dev_every_steps"]
                        == 0
                    ):
                        if self.accelerator.is_main_process:
                            logger.info(
                                f"Running evaluation at step {self.global_step}..."
                            )
                            recall = self._evaluate()
                            logger.info(
                                f"Step {self.global_step}: dev_recall@{self.config['dev']['recall_k']}={recall:.4f}"
                            )
                            wandb.log(
                                {"global_step": self.global_step, "dev_recall": recall}
                            )

                            if recall > self.best_recall:
                                self.best_recall = recall
                                self._save_checkpoint("best")

                        self.accelerator.wait_for_everyone()
                        self.encoder.train()

            pbar.close()

        self._save_checkpoint("final")
        if self.accelerator.is_main_process:
            wandb.finish()


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    trainer = GRAFTTrainer(sys.argv[1])
    trainer.train()
