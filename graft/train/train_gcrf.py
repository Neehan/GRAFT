"""Main training loop for GRAFT with Accelerate, wandb, and hard-negative mining."""

import logging
import yaml
import torch
import wandb
from pathlib import Path
from tqdm import tqdm
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset

from graft.models.encoder import Encoder
from graft.models.gnn import GraphSAGE
from graft.train.losses import compute_total_loss
from graft.train.sampler import GraphBatchSampler
from graft.train.mining import HardNegativeMiner
from graft.data.pair_maker import load_query_pairs

logger = logging.getLogger("graft.train")


def train(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    accelerator = Accelerator(mixed_precision="bf16" if cfg["train"]["bf16"] else "no")
    device = accelerator.device

    wandb.init(
        project="graft",
        name=cfg["experiment"]["name"],
        config=cfg
    )
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")

    encoder = Encoder(
        model_name=cfg["encoder"]["model_name"],
        max_len=cfg["encoder"]["max_len"],
        pool=cfg["encoder"]["pool"],
        proj_dim=cfg["encoder"]["proj_dim"],
        freeze_layers=cfg["encoder"]["freeze_layers"]
    )

    gnn = GraphSAGE(
        in_dim=cfg["gnn"]["hidden_dim"],
        hidden_dim=cfg["gnn"]["hidden_dim"],
        layers=cfg["gnn"]["layers"],
        dropout=cfg["gnn"]["dropout"]
    )

    logger.info("Loading HotpotQA datasets...")
    graph = torch.load(cfg["data"]["graph_path"], weights_only=False)
    train_dataset = load_dataset("hotpot_qa", "distractor", split="train")
    dev_dataset = load_dataset("hotpot_qa", "distractor", split="validation")

    train_pairs = load_query_pairs(train_dataset, cfg["data"]["graph_path"])
    dev_pairs = load_query_pairs(dev_dataset, cfg["data"]["graph_path"])

    sampler = GraphBatchSampler(
        graph=graph,
        train_pairs=train_pairs,
        batch_size_queries=cfg["train"]["batch_size_queries"],
        fanouts=cfg["gnn"]["fanouts"]
    )

    miner = HardNegativeMiner(encoder, device)

    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": cfg["train"]["lr_encoder"]},
        {"params": gnn.parameters(), "lr": cfg["train"]["lr_gnn"]}
    ], weight_decay=cfg["train"]["weight_decay"])

    total_steps = len(sampler) * cfg["train"]["epochs"]
    warmup_steps = int(total_steps * cfg["train"]["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    encoder, gnn, optimizer, scheduler = accelerator.prepare(encoder, gnn, optimizer, scheduler)

    global_step = 0
    best_recall = 0.0

    for epoch in range(cfg["train"]["epochs"]):
        encoder.train()
        gnn.train()

        pbar = tqdm(sampler, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")

        for batch in pbar:
            queries = batch["queries"]
            pos_nodes = batch["pos_nodes"]
            neg_nodes = batch["neg_nodes"]
            subgraph = batch["subgraph"]

            if global_step > 0 and global_step % cfg["train"]["hardneg_refresh_steps"] == 0:
                logger.info(f"Refreshing hard negatives at step {global_step}")
                miner.build_index(graph.node_text)
                neg_nodes = miner.mine_hard_negatives(queries, pos_nodes, k=5)

            query_encoded = encoder.tokenizer(
                queries,
                max_length=cfg["encoder"]["max_len"],
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            query_embeds = encoder(query_encoded["input_ids"], query_encoded["attention_mask"])

            subgraph_node_ids = subgraph.n_id
            subgraph_texts = [graph.node_text[int(nid)] for nid in subgraph_node_ids.cpu().numpy()]

            node_encoded = encoder.tokenizer(
                subgraph_texts,
                max_length=cfg["encoder"]["max_len"],
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            node_embeds_raw = encoder(node_encoded["input_ids"], node_encoded["attention_mask"])

            node_embeds_gnn = gnn(node_embeds_raw, subgraph.edge_index.to(device))

            pos_indices_in_subgraph = torch.tensor(
                [torch.where(subgraph_node_ids == pid)[0][0].item() for pid in pos_nodes],
                device=device
            )

            labels = pos_indices_in_subgraph

            loss, loss_q2d, loss_nbr = compute_total_loss(
                query_embeds=query_embeds,
                doc_embeds=node_embeds_gnn,
                labels=labels,
                node_embeds=node_embeds_gnn,
                edge_index=subgraph.edge_index.to(device),
                pos_edges=None,
                neg_edges=None,
                lambda_q2d=cfg["loss"]["lambda_q2d"],
                tau=cfg["loss"]["tau"],
                tau_graph=cfg["loss"]["tau_graph"],
                alpha_link=cfg["loss"]["alpha_link"]
            )

            accelerator.backward(loss)

            if cfg["train"]["grad_clip"] > 0:
                accelerator.clip_grad_norm_(
                    list(encoder.parameters()) + list(gnn.parameters()),
                    cfg["train"]["grad_clip"]
                )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % cfg["train"]["log_every"] == 0:
                metrics = {
                    "global_step": global_step,
                    "loss": loss.item(),
                    "loss_q2d": loss_q2d.item(),
                    "loss_nbr": loss_nbr.item(),
                    "lr_encoder": scheduler.get_last_lr()[0],
                    "lr_gnn": scheduler.get_last_lr()[1],
                }
                wandb.log(metrics)
                logger.info(f"Step {global_step}: loss={loss.item():.4f}, loss_q2d={loss_q2d.item():.4f}, loss_nbr={loss_nbr.item():.4f}")

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})

            if global_step % cfg["train"]["eval_every_steps"] == 0:
                logger.info(f"Running evaluation at step {global_step}...")
                recall = evaluate(encoder, dev_pairs, cfg, device)
                logger.info(f"Step {global_step}: dev_recall@10={recall:.4f}")
                wandb.log({"global_step": global_step, "dev_recall@10": recall})

                if recall > best_recall:
                    best_recall = recall
                    save_checkpoint(encoder, gnn, cfg["experiment"]["output_dir"], "best")

                encoder.train()
                gnn.train()

    save_checkpoint(encoder, gnn, cfg["experiment"]["output_dir"], "final")
    wandb.finish()


def evaluate(encoder, dev_pairs, cfg, device):
    encoder.eval()

    graph = torch.load(cfg["data"]["graph_path"], weights_only=False)
    num_nodes = len(graph.node_text)

    correct = 0
    total = min(len(dev_pairs), 100)

    with torch.no_grad():
        for i in range(total):
            pair = dev_pairs[i]
            query = pair["query"]
            pos_node = pair["pos_node"]

            query_embed = encoder.encode([query], device)

            neg_indices = torch.randint(0, num_nodes, (10,))
            candidate_indices = torch.cat([torch.tensor([pos_node]), neg_indices])

            candidate_texts = [graph.node_text[int(idx)] for idx in candidate_indices]
            candidate_embeds = encoder.encode(candidate_texts, device)

            scores = torch.matmul(query_embed, candidate_embeds.T).squeeze(0)
            top_k_indices = torch.topk(scores, k=min(10, len(scores))).indices

            if 0 in top_k_indices:
                correct += 1

    recall_at_10 = correct / total
    return recall_at_10


def save_checkpoint(encoder, gnn, output_dir, tag):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), f"{output_dir}/encoder_{tag}.pt")
    torch.save(gnn.state_dict(), f"{output_dir}/gnn_{tag}.pt")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    train(sys.argv[1])
