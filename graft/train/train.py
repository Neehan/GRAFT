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
from graft.models.gnn import GraphSAGE
from graft.train.losses import compute_total_loss
from graft.train.sampler import GraphBatchSampler
from graft.data.pair_maker import load_query_pairs

logger = logging.getLogger("graft.train")


class GRAFTTrainer:
    def __init__(self, config_path):
        self.cfg_path = config_path
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        # Set seed for reproducibility
        seed = self.cfg["train"]["seed"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg["train"][
                "gradient_accumulation_steps"
            ],
            kwargs_handlers=[ddp_kwargs],
        )
        self.device = self.accelerator.device

        if self.accelerator.is_main_process:
            wandb.init(
                project="graft",
                name=self.cfg["experiment"]["name"],
                config=self.cfg,
            )
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step")

        self._setup_models()
        self._setup_data()
        self._setup_training()

        self.global_step: int = 0
        self.best_recall: float = 0.0

    def _get_graph_path(self):
        """Get path to prepared graph."""
        graph_dir = Path(self.cfg["data"]["graph_dir"])
        graph_name = self.cfg["data"]["graph_name"]
        semantic_k = self.cfg["data"].get("semantic_k")
        knn_only = self.cfg["data"].get("knn_only", False)

        if semantic_k is None:
            graph_path = graph_dir / f"{graph_name}.pt"
        else:
            suffix = f"_knn_only{semantic_k}" if knn_only else f"_knn{semantic_k}"
            graph_path = graph_dir / f"{graph_name}{suffix}.pt"

        if not graph_path.exists():
            raise FileNotFoundError(
                f"Graph not found: {graph_path}\n"
                f"Run data preparation first: bash scripts/prepare_data.sh {self.cfg_path}"
            )

        if self.accelerator.is_main_process:
            logger.info(f"Using graph: {graph_path}")
        return str(graph_path)

    def _setup_models(self):
        self.encoder = Encoder(
            model_name=self.cfg["encoder"]["model_name"],
            max_len=self.cfg["encoder"]["max_len"],
            pool=self.cfg["encoder"]["pool"],
            freeze_layers=self.cfg["encoder"]["freeze_layers"],
        )

        self.tokenizer = self.encoder.tokenizer

        self.gnn = GraphSAGE(
            in_dim=self.cfg["gnn"]["hidden_dim"],
            hidden_dim=self.cfg["gnn"]["hidden_dim"],
            layers=self.cfg["gnn"]["layers"],
            dropout=self.cfg["gnn"]["dropout"],
        )

    def _setup_data(self):
        if self.accelerator.is_main_process:
            logger.info("Loading mteb/hotpotqa datasets...")
        graph_path = self._get_graph_path()
        self.graph = torch.load(graph_path, weights_only=False)

        train_pairs = load_query_pairs(
            "train", graph_path, self.cfg, log=self.accelerator.is_main_process
        )

        with self.accelerator.main_process_first():
            self.sampler = GraphBatchSampler(
                graph=self.graph,
                train_pairs=train_pairs,
                query_batch_size=self.cfg["train"]["query_batch_size"],
                fanouts=self.cfg["gnn"]["fanouts"],
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
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.encoder.parameters(),
                    "lr": self.cfg["train"]["lr_encoder"],
                },
                {"params": self.gnn.parameters(), "lr": self.cfg["train"]["lr_gnn"]},
            ],
            weight_decay=self.cfg["train"]["weight_decay"],
        )

        total_steps = len(self.sampler) * self.cfg["train"]["epochs"]
        warmup_steps = int(total_steps * self.cfg["train"]["warmup_ratio"])
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        self.encoder, self.gnn, self.optimizer, self.scheduler = (
            self.accelerator.prepare(self.encoder, self.gnn, optimizer, scheduler)
        )

    def _encode_texts(self, texts):
        """Tokenize and encode texts in one forward pass."""
        encoded = self.tokenizer(
            texts,
            max_length=self.cfg["encoder"]["max_len"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(self.device)
        mask = encoded["attention_mask"].to(self.device)
        return self.encoder(ids, mask)

    def _load_fixed_dev_set(self):
        """Load pre-built fixed dev set from disk."""
        graph_dir = Path(self.cfg["data"]["graph_dir"])
        graph_name = self.cfg["data"]["graph_name"]
        semantic_k = self.cfg["data"].get("semantic_k")
        knn_only = self.cfg["data"].get("knn_only", False)

        # Construct dev set filename matching the graph
        if semantic_k is None:
            dev_set_name = f"{graph_name}_dev_set.pt"
        else:
            suffix = f"_knn_only{semantic_k}" if knn_only else f"_knn{semantic_k}"
            dev_set_name = f"{graph_name}{suffix}_dev_set.pt"

        dev_set_path = graph_dir / dev_set_name

        if not dev_set_path.exists():
            raise FileNotFoundError(
                f"Dev set not found: {dev_set_path}\n"
                f"Run data preparation first: bash scripts/prepare_data.sh {self.cfg_path}"
            )

        # Load dev set with candidate indices
        dev_set_raw = torch.load(dev_set_path, weights_only=False)

        # Convert indices to texts
        dev_set = []
        for item in dev_set_raw:
            candidate_texts = [
                self.graph.node_text[int(idx)] for idx in item["candidate_indices"]
            ]
            dev_set.append(
                {
                    "query": item["query"],
                    "candidate_texts": candidate_texts,
                    "num_positives": item["num_positives"],
                }
            )

        if self.accelerator.is_main_process:
            logger.info(
                f"Loaded small dev set from {dev_set_path}: {len(dev_set)} queries"
            )
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

        # GNN
        node_embeds = self.gnn(
            node_embeds, batch["subgraph"].edge_index.to(self.device)
        )

        # Labels: Sample 1 positive per query to avoid averaging (which pulls query to centroid)
        # batch["pos_nodes"] is a list of lists: [[pos1_1, pos1_2], [pos2_1], ...]
        subgraph_ids = batch["subgraph"].n_id.to(self.device)

        labels = []
        for pos_nodes in batch["pos_nodes"]:
            pos_tensor = torch.tensor(pos_nodes, device=self.device, dtype=torch.long)
            matches = (subgraph_ids.unsqueeze(0) == pos_tensor.unsqueeze(1)).any(dim=0)
            pos_indices = matches.nonzero(as_tuple=False).squeeze(-1)

            # Randomly sample 1 positive (different each step for data augmentation)
            if len(pos_indices) > 0:
                random_idx = torch.randint(0, len(pos_indices), (1,)).item()
                labels.append(pos_indices[random_idx])
            else:
                raise ValueError("No positives found for query")

        labels = torch.stack(labels)

        # Loss
        loss, loss_q2d, loss_nbr = compute_total_loss(
            query_embeds=query_embeds,
            doc_embeds=node_embeds,
            labels=labels,
            node_embeds=node_embeds,
            edge_index=batch["subgraph"].edge_index.to(self.device),
            pos_edges=(
                batch.get("pos_edges").to(self.device)
                if batch.get("pos_edges") is not None
                else None
            ),
            neg_edges=(
                batch.get("neg_edges").to(self.device)
                if batch.get("neg_edges") is not None
                else None
            ),
            **{
                k: self.cfg["loss"][k]
                for k in ["lambda_q2d", "tau", "tau_graph", "alpha_link"]
            },
        )

        return {"loss": loss, "loss_q2d": loss_q2d, "loss_nbr": loss_nbr}

    def _evaluate(self):
        """Dev set evaluation during training for monitoring progress."""
        self.encoder.eval()

        unwrapped_encoder = self.accelerator.unwrap_model(self.encoder)

        correct = 0
        total = len(self.dev_data)
        recall_k = self.cfg["eval"]["recall_k"]
        query_batch_size = self.cfg["eval"]["query_batch_size"]

        with torch.no_grad():
            for batch_start in tqdm(
                range(0, total, query_batch_size),
                desc="Evaluating",
                total=(total + query_batch_size - 1) // query_batch_size,
            ):
                batch_end = min(batch_start + query_batch_size, total)
                batch_items = self.dev_data[batch_start:batch_end]
                batch_size = len(batch_items)

                queries = [item["query"] for item in batch_items]
                query_embeds = unwrapped_encoder.encode(queries, self.device)

                # All queries have same number of candidates now (fixed in _load_fixed_dev_set)
                all_candidates = []
                for item in batch_items:
                    all_candidates.extend(item["candidate_texts"])

                # Encode all candidates at once
                all_candidate_embeds = unwrapped_encoder.encode(
                    all_candidates, self.device
                )

                # Reshape into [batch_size, num_candidates, hidden_dim]
                num_candidates = len(batch_items[0]["candidate_texts"])
                candidate_embeds = all_candidate_embeds.reshape(
                    batch_size, num_candidates, -1
                )

                scores = torch.bmm(
                    query_embeds.unsqueeze(1), candidate_embeds.transpose(1, 2)
                ).squeeze(1)

                top_k_indices = torch.topk(
                    scores, k=min(recall_k, scores.size(1)), dim=1
                ).indices

                for j, item in enumerate(batch_items):
                    num_pos = item["num_positives"]
                    pos_indices = set(range(num_pos))
                    retrieved_indices = set(top_k_indices[j].cpu().tolist())
                    if pos_indices & retrieved_indices:
                        correct += 1

        return correct / total

    def _save_checkpoint(self, tag):
        """Save model checkpoints."""
        if self.accelerator.is_main_process:
            output_dir = Path(self.cfg["experiment"]["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            unwrapped_encoder = self.accelerator.unwrap_model(self.encoder)
            unwrapped_gnn = self.accelerator.unwrap_model(self.gnn)
            torch.save(unwrapped_encoder.state_dict(), output_dir / f"encoder_{tag}.pt")
            torch.save(unwrapped_gnn.state_dict(), output_dir / f"gnn_{tag}.pt")

    def train(self):
        """Main training loop."""
        if self.accelerator.is_main_process:
            logger.info("Running zero-shot evaluation...")
            zero_shot_recall = self._evaluate()
            logger.info(
                f"Zero-shot: dev_recall@{self.cfg['eval']['recall_k']}={zero_shot_recall:.4f}"
            )
            wandb.log({"global_step": 0, "dev_recall": zero_shot_recall})

        self.accelerator.wait_for_everyone()

        for epoch in range(self.cfg["train"]["epochs"]):
            self.encoder.train()
            self.gnn.train()

            total_steps = (
                len(self.sampler) // self.cfg["train"]["gradient_accumulation_steps"]
            )
            pbar = tqdm(
                total=total_steps, desc=f"Epoch {epoch+1}/{self.cfg['train']['epochs']}"
            )

            for batch in self.sampler:
                with self.accelerator.accumulate(self.encoder, self.gnn):
                    step_output = self._training_step(batch)
                    loss = step_output["loss"]
                    loss_q2d = step_output["loss_q2d"]
                    loss_nbr = step_output["loss_nbr"]

                    self.accelerator.backward(loss)

                    if self.cfg["train"]["grad_clip"] > 0:
                        self.accelerator.clip_grad_norm_(
                            list(self.encoder.parameters())
                            + list(self.gnn.parameters()),
                            self.cfg["train"]["grad_clip"],
                        )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    if self.accelerator.sync_gradients:
                        self.global_step += 1

                    if (
                        self.accelerator.sync_gradients
                        and self.global_step % self.cfg["train"]["log_every"] == 0
                    ):
                        if self.accelerator.is_main_process:
                            metrics = {
                                "global_step": self.global_step,
                                "loss": loss.item(),
                                "loss_q2d": loss_q2d.item(),
                                "loss_nbr": loss_nbr.item(),
                                "lr_encoder": self.scheduler.get_last_lr()[0],
                                "lr_gnn": self.scheduler.get_last_lr()[1],
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
                        and self.global_step % self.cfg["train"]["eval_every_steps"]
                        == 0
                    ):
                        if self.accelerator.is_main_process:
                            logger.info(
                                f"Running evaluation at step {self.global_step}..."
                            )
                            recall = self._evaluate()
                            logger.info(
                                f"Step {self.global_step}: dev_recall@{self.cfg['eval']['recall_k']}={recall:.4f}"
                            )
                            wandb.log(
                                {"global_step": self.global_step, "dev_recall": recall}
                            )

                            if recall > self.best_recall:
                                self.best_recall = recall
                                self._save_checkpoint("best")

                        self.accelerator.wait_for_everyone()
                        self.encoder.train()
                        self.gnn.train()

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
