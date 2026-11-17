import os, sys, time, json, argparse
from types import SimpleNamespace
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.transforms.GMPE.builder import build_pretrain_system  
from src.utils.multiview import choose_device


def parse_args():
    p = argparse.ArgumentParser("GMPE pretrain (graph alignment)")
    
    p.add_argument("--dataset", type=str, required=True)

    p.add_argument("--pcqm_root", type=str, default=None)

    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--avoid_disconnect", action="store_true")
    
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--sim", type=str, default="dot", choices=["dot", "cos"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--normalize_emb", action="store_true")
    
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])
    
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="checkpoints/gmpe")
    p.add_argument("--log_dir", type=str, default="logs/gmpe")
    
    p.add_argument("--csv_path", type=str, default=None)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--split_seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ckpt_dir = os.path.join(args.save_dir, run_name); os.makedirs(ckpt_dir, exist_ok=True)
    log_dir = os.path.join(args.log_dir, run_name); os.makedirs(log_dir, exist_ok=True)
    if not distributed or int(os.environ.get("RANK", "0")) == 0:
        with open(os.path.join(ckpt_dir, "hparams.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        device = choose_device(args.device)
    is_main = not distributed or dist.get_rank() == 0
    system: SimpleNamespace = build_pretrain_system(args, device)
    if distributed:
        def shard_loader(loader, shuffle):
            sampler = DistributedSampler(loader.dataset, num_replicas=world_size,
                                         rank=dist.get_rank(), shuffle=shuffle,
                                         seed=args.seed, drop_last=False)
            return DataLoader(loader.dataset, batch_size=loader.batch_size, sampler=sampler,
                              num_workers=loader.num_workers, pin_memory=loader.pin_memory,
                              collate_fn=loader.collate_fn, drop_last=loader.drop_last)
        system.train_loader = shard_loader(system.train_loader, True)
        system.val_loader = shard_loader(system.val_loader, False)
        system.model = DDP(system.model, device_ids=[local_rank], output_device=local_rank,
                             find_unused_parameters=True)
        
        raw_model = system.model.module
        system.criterion = lambda outputs, batch: raw_model.align_loss(
            outputs["Sigma_list"], [p.to(outputs["X"].device) for p in batch["perm"]])

    model      = system.model
    optimizer  = system.optimizer
    scheduler  = system.scheduler
    train_loader = system.train_loader
    val_loader   = system.val_loader
    criterion    = system.criterion
    infer_perm   = system.infer_perm
    eval_align   = system.eval_align

    tb = SummaryWriter(log_dir) if is_main else None

    best_acc = 0.0
    best_path = os.path.join(ckpt_dir, "best.ckpt")
    checkpoint_args = vars(args).copy()
    checkpoint_args.pop("dataset", None)

    for epoch in range(1, args.epochs + 1):
        if distributed:
            system.train_loader.sampler.set_epoch(epoch - 1)
            system.val_loader.sampler.set_epoch(epoch - 1)
        
        model.train()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for batch in pbar:
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

            outputs = model(batch)
            loss = criterion(outputs, batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix({"loss": f"{running_loss / n_batches:.4f}"})

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / max(1, n_batches)
        if tb:
            tb.add_scalar("train/loss", train_loss, epoch)

        
        model.eval()
        total_hits, total_nodes = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
                pred_perm = infer_perm(model, batch)
                hits, nodes = eval_align(pred_perm, batch["perm"])
                total_hits += hits
                total_nodes += nodes

        if distributed:
            counts = torch.tensor([total_hits, total_nodes], dtype=torch.long, device=device)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            total_hits, total_nodes = map(int, counts.tolist())
        val_acc = total_hits / max(1, total_nodes)
        if tb:
            tb.add_scalar("val/align_acc", val_acc, epoch)
            tb.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        
        if (epoch == 1 or val_acc > best_acc) and is_main:
            best_acc = val_acc
            torch.save(
                {
                    "model": (model.module if distributed else model).state_dict(),
                    "args": checkpoint_args,
                },
                best_path,
            )

        if is_main:
            print(f"[Epoch {epoch}] train_loss={train_loss:.4f}  val_align_acc={val_acc:.4f}  best={best_acc:.4f}")

    if is_main:
        print(f"[GMPE] Done. Best val align acc = {best_acc:.4f}  -> {best_path}")
    if tb:
        tb.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
