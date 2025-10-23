"""Main training loop for GRAFT with Accelerate, wandb, and hard-negative mining."""

import logging
import warnings
import yaml
import torch
import wandb
from pathlib import Path
from tqdm import tqdm
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
import os

os.environ["PYTHONWARNINGS"] = "ignore"

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

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg["train"][
                "gradient_accumulation_steps"
            ],
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
            "train", graph_path, log=self.accelerator.is_main_process
        )
        dev_pairs = load_query_pairs(
            "dev", graph_path, log=self.accelerator.is_main_process
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
            logger.info("Building fixed eval set with sampled negatives...")
        self.eval_data = self._build_fixed_eval_set(dev_pairs)

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

    def _tokenize(self, texts):
        """DRY helper for tokenization."""
        return self.tokenizer(
            texts,
            max_length=self.cfg["encoder"]["max_len"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

    def _build_fixed_eval_set(self, dev_pairs):
        """Build fixed evaluation set with pre-sampled negatives for consistency."""
        num_nodes = len(self.graph.node_text)
        num_samples = min(len(dev_pairs), self.cfg["eval"]["num_samples"])
        num_negatives = self.cfg["eval"]["num_negatives"]

        eval_set = []
        for i in range(num_samples):
            pair = dev_pairs[i]
            pos_node = pair["pos_node"]

            neg_indices = torch.randint(0, num_nodes, (num_negatives,))
            candidate_indices = torch.cat([torch.tensor([pos_node]), neg_indices])
            candidate_texts = [
                self.graph.node_text[int(idx)] for idx in candidate_indices
            ]

            eval_set.append(
                {
                    "query": pair["query"],
                    "candidate_texts": candidate_texts,
                }
            )

        if self.accelerator.is_main_process:
            logger.info(
                f"Built fixed eval set: {len(eval_set)} queries, {num_negatives} negatives each"
            )
        return eval_set

    def _training_step(self, batch):
        """Single training step."""
        queries = batch["queries"]
        pos_nodes = batch["pos_nodes"]
        subgraph = batch["subgraph"]

        query_encoded = self._tokenize(queries)
        query_embeds = self.encoder(
            query_encoded["input_ids"].clone(), query_encoded["attention_mask"].clone()
        )

        subgraph_node_ids = subgraph.n_id
        subgraph_texts = [
            self.graph.node_text[int(nid)] for nid in subgraph.n_id_cpu.numpy()
        ]

        node_encoded = self._tokenize(subgraph_texts)

        # Batch encoding to avoid OOM with large subgraphs
        encoder_batch_size = self.cfg["encoder"]["batch_size"]
        node_embeds_list = []
        num_nodes = node_encoded["input_ids"].size(0)

        for i in range(0, num_nodes, encoder_batch_size):
            batch_input_ids = node_encoded["input_ids"][
                i : i + encoder_batch_size
            ].clone()
            batch_attention_mask = node_encoded["attention_mask"][
                i : i + encoder_batch_size
            ].clone()
            batch_embeds = self.encoder(batch_input_ids, batch_attention_mask)
            node_embeds_list.append(batch_embeds)

        node_embeds_raw = torch.cat(node_embeds_list, dim=0)

        edge_index_gpu = subgraph.edge_index.clone().to(self.device)
        node_embeds_gnn = self.gnn(node_embeds_raw, edge_index_gpu)

        # Vectorized index lookup - avoid Python loop with .item() syncs
        pos_nodes_tensor = pos_nodes.clone().to(self.device)
        subgraph_node_ids_gpu = subgraph_node_ids.clone().to(self.device)
        pos_indices_in_subgraph = (
            (subgraph_node_ids_gpu.unsqueeze(1) == pos_nodes_tensor.unsqueeze(0))
            .nonzero()[:, 0]
            .clone()
        )

        labels = pos_indices_in_subgraph

        pos_edges_gpu = (
            batch.get("pos_edges").clone().to(self.device)
            if batch.get("pos_edges") is not None
            else None
        )
        neg_edges_gpu = (
            batch.get("neg_edges").clone().to(self.device)
            if batch.get("neg_edges") is not None
            else None
        )

        loss, loss_q2d, loss_nbr = compute_total_loss(
            query_embeds=query_embeds,
            doc_embeds=node_embeds_gnn,
            labels=labels,
            node_embeds=node_embeds_gnn,
            edge_index=edge_index_gpu,
            pos_edges=pos_edges_gpu,
            neg_edges=neg_edges_gpu,
            lambda_q2d=self.cfg["loss"]["lambda_q2d"],
            tau=self.cfg["loss"]["tau"],
            tau_graph=self.cfg["loss"]["tau_graph"],
            alpha_link=self.cfg["loss"]["alpha_link"],
        )

        return {"loss": loss, "loss_q2d": loss_q2d, "loss_nbr": loss_nbr}

    def _evaluate(self):
        """Fast training evaluation: 1 pos + random negatives (proxy for overfitting detection)."""
        self.encoder.eval()

        unwrapped_encoder = self.accelerator.unwrap_model(self.encoder)

        correct = 0
        total = len(self.eval_data)
        recall_k = self.cfg["eval"]["recall_k"]
        query_batch_size = self.cfg["eval"]["query_batch_size"]

        with torch.no_grad():
            for batch_start in tqdm(
                range(0, total, query_batch_size),
                desc="Evaluating",
                total=(total + query_batch_size - 1) // query_batch_size,
            ):
                batch_end = min(batch_start + query_batch_size, total)
                batch_items = self.eval_data[batch_start:batch_end]
                batch_size = len(batch_items)

                queries = [item["query"] for item in batch_items]
                query_embeds = unwrapped_encoder.encode(queries, self.device)

                all_candidates = []
                num_candidates_per_query = len(batch_items[0]["candidate_texts"])
                for item in batch_items:
                    all_candidates.extend(item["candidate_texts"])

                # Sub-batch candidates to avoid OOM - use encoder batch size
                candidate_batch_size = self.cfg["encoder"]["batch_size"]
                candidate_embeds_list = []
                for i in range(0, len(all_candidates), candidate_batch_size):
                    batch = all_candidates[i : i + candidate_batch_size]
                    batch_embeds = unwrapped_encoder.encode(batch, self.device)
                    candidate_embeds_list.append(batch_embeds)

                candidate_embeds = torch.cat(candidate_embeds_list, dim=0)
                candidate_embeds = candidate_embeds.reshape(
                    batch_size, num_candidates_per_query, -1
                )

                scores = torch.bmm(
                    query_embeds.unsqueeze(1), candidate_embeds.transpose(1, 2)
                ).squeeze(1)

                top_k_indices = torch.topk(
                    scores, k=min(recall_k, scores.size(1)), dim=1
                ).indices

                correct += (top_k_indices == 0).any(dim=1).sum().item()

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
            # zero_shot_recall = self._evaluate()
            # logger.info(
            #     f"Zero-shot: dev_recall@{self.cfg['eval']['recall_k']}={zero_shot_recall:.4f}"
            # )
            # wandb.log({"global_step": 0, "dev_recall": zero_shot_recall})

        self.accelerator.wait_for_everyone()

        for epoch in range(self.cfg["train"]["epochs"]):
            self.encoder.train()
            self.gnn.train()

            pbar = tqdm(
                self.sampler, desc=f"Epoch {epoch+1}/{self.cfg['train']['epochs']}"
            )

            for batch in pbar:
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
