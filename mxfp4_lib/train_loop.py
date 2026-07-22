"""Shared causal LM training loop (FineWeb-Edu + optional DDP + val loss + resume)."""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from mxfp4_lib.data import build_train_loader, build_val_loader, steps_for_target_tokens
from mxfp4_lib.util import append_jsonl, cosine_lr, ensure_dir


def _ddp_setup():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, torch.device(f"cuda:{local_rank}")


def _ddp_cleanup(enabled: bool):
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def _raw_model(model):
    return model.module if hasattr(model, "module") else model


def _resume_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / "resume" / "train_state.pt"


def _latest_hf_dir(out_dir: str | Path) -> Path:
    return Path(out_dir) / "resume" / "hf_latest"


@torch.no_grad()
def eval_val_loss(model, loader, device, use_bf16: bool) -> float:
    was_training = model.training
    model.eval()
    raw = _raw_model(model)
    nll = 0.0
    ntok = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = raw(input_ids=input_ids, labels=labels)
        valid = labels.numel()
        nll += float(out.loss.item()) * valid
        ntok += valid
        if ntok > 2_000_000:
            break
    if was_training:
        model.train()
    return nll / max(1, ntok)


def save_training_checkpoint(
    *,
    model,
    opt,
    tok,
    out_dir: str | Path,
    step: int,
    max_steps: int,
    last_logged_loss,
    best_val,
    last_val,
    t0: float,
    cfg: dict,
    pre_save=None,
    tag: str = "periodic",
) -> Path:
    """
    Persist resumable state under out_dir/resume/.
    Rank 0 only. Does not move the live training model off GPU.
    """
    del pre_save  # reserved; FQ models resume via state_dict
    out_dir = Path(out_dir)
    resume_dir = ensure_dir(out_dir / "resume")
    raw = _raw_model(model)

    state = {
        "step": step,
        "max_steps": max_steps,
        "optimizer": opt.state_dict(),
        "last_logged_loss": last_logged_loss,
        "best_val": best_val,
        "last_val": last_val,
        "elapsed": time.time() - t0,
        "seed": cfg.get("seed"),
        "tag": tag,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    state_path = resume_dir / "train_state.pt"
    tmp_state = resume_dir / "train_state.pt.tmp"
    torch.save(state, tmp_state)
    tmp_state.replace(state_path)

    sd = {k: v.detach().cpu() for k, v in raw.state_dict().items()}
    ms_tmp = resume_dir / "model_state.pt.tmp"
    torch.save(sd, ms_tmp)
    ms_tmp.replace(resume_dir / "model_state.pt")

    try:
        tok.save_pretrained(resume_dir / "tokenizer")
    except Exception as e:
        print(f"[ckpt] tokenizer save skipped: {e}", flush=True)

    meta = {
        "step": step,
        "max_steps": max_steps,
        "tag": tag,
        "path": str(state_path),
        "seconds": time.time() - t0,
    }
    with open(resume_dir / "checkpoint_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[ckpt] saved step={step}/{max_steps} → {state_path}", flush=True)
    return state_path


def load_training_checkpoint(out_dir: str | Path, model, opt, device) -> dict | None:
    path = _resume_path(out_dir)
    if not path.exists():
        # Also accept model_state-only recovery
        ms = Path(out_dir) / "resume" / "model_state.pt"
        if ms.exists():
            sd = torch.load(ms, map_location="cpu", weights_only=True)
            _raw_model(model).load_state_dict(sd, strict=False)
            print(f"[ckpt] loaded weights-only from {ms} (no optimizer/step)")
            return {"step": 0, "weights_only": True}
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    ms = Path(out_dir) / "resume" / "model_state.pt"
    if ms.exists():
        sd = torch.load(ms, map_location="cpu", weights_only=True)
        _raw_model(model).load_state_dict(sd, strict=False)
    opt.load_state_dict(blob["optimizer"])
    # move optimizer state to device
    for state in opt.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    if blob.get("rng"):
        try:
            torch.set_rng_state(blob["rng"]["torch"])
            if blob["rng"].get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(blob["rng"]["cuda"])
        except Exception as e:
            print(f"[ckpt] rng restore skipped: {e}")
    print(f"[ckpt] resumed from step={blob['step']} ({path})", flush=True)
    return blob


def train_loop(
    model: AutoModelForCausalLM,
    cfg: dict,
    out_dir: str,
    log_name: str,
    max_steps: int | None = None,
    pre_save=None,
) -> dict:
    """Train for optimizer steps. Supports torchrun DDP, periodic ckpt, resume."""
    ddp, rank, world_size, device = _ddp_setup()
    is_main = rank == 0

    model.to(device)
    model.train()
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], find_unused_parameters=False
        )

    tok_src = cfg["paths"].get("init_model") or str(Path(cfg["paths"]["data_dir"]) / "tokenizer")
    tok = AutoTokenizer.from_pretrained(tok_src)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if ddp:
        if is_main:
            build_train_loader(cfg, tok, rank=0, world_size=1)
            if int(cfg.get("eval", {}).get("val_tokens") or 0) > 0:
                build_val_loader(cfg, tok)
        dist.barrier()

    loader = build_train_loader(cfg, tok, rank=rank, world_size=world_size)
    val_loader = build_val_loader(cfg, tok) if is_main else None

    tcfg = cfg["train"]
    if max_steps is None:
        if tcfg.get("target_tokens"):
            max_steps = steps_for_target_tokens(cfg, world_size=world_size)
        else:
            max_steps = int(tcfg["max_steps"])
    accum = int(tcfg["grad_accum"])
    save_every = int(tcfg.get("save_every") or 0)
    resume_enabled = bool(tcfg.get("resume", True))

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
        betas=(0.9, 0.95),
    )

    log_path = Path(cfg["paths"]["results"]) / log_name
    ensure_dir(log_path.parent)

    use_bf16 = bool(tcfg.get("bf16", True)) and device.type == "cuda"
    val_every = int(cfg.get("eval", {}).get("val_every_steps") or 0)
    step = 0
    log_loss_sum = 0.0
    log_count = 0
    t0 = time.time()
    elapsed_offset = 0.0
    last_logged_loss = None
    best_val = None
    last_val = None

    if resume_enabled:
        # Every rank loads the same resume files (model + opt); step taken from blob
        blob = load_training_checkpoint(out_dir, model, opt, device)
        if blob and not blob.get("weights_only"):
            step = int(blob["step"])
            last_logged_loss = blob.get("last_logged_loss")
            best_val = blob.get("best_val")
            last_val = blob.get("last_val")
            elapsed_offset = float(blob.get("elapsed") or 0.0)
        if ddp:
            step_t = torch.tensor([step], device=device, dtype=torch.long)
            dist.broadcast(step_t, src=0)
            step = int(step_t.item())
            dist.barrier()

    data_iter = iter(loader)
    pbar = tqdm(total=max_steps, initial=step, desc=f"train->{Path(out_dir).name}", disable=not is_main)
    opt.zero_grad(set_to_none=True)

    # Graceful checkpoint on SIGTERM/SIGINT (slurm / machine reclaim)
    _save_request = {"flag": False}

    def _handle_signal(signum, _frame):
        print(f"[ckpt] signal {signum} → will checkpoint after this step", flush=True)
        _save_request["flag"] = True

    if is_main:
        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

    if is_main:
        print(
            f"[train] steps={max_steps} start_step={step} world_size={world_size} "
            f"save_every={save_every} target_tokens={tcfg.get('target_tokens')} "
            f"seq={cfg['data']['seq_len']}",
            flush=True,
        )

    while step < max_steps:
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(step)

        lr = cosine_lr(step, tcfg["warmup_steps"], max_steps, tcfg["lr"], tcfg["min_lr"])
        for g in opt.param_groups:
            g["lr"] = lr

        micro_losses = []
        for _ in range(accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                out = model(input_ids=input_ids, labels=labels)
                loss = out.loss / accum
            loss.backward()
            micro_losses.append(float(out.loss.detach().item()))

        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
        opt.step()
        opt.zero_grad(set_to_none=True)

        step += 1
        step_loss = sum(micro_losses) / len(micro_losses)
        log_loss_sum += step_loss
        log_count += 1
        pbar.update(1)

        if step % int(tcfg["log_every"]) == 0 and is_main:
            avg = log_loss_sum / max(1, log_count)
            last_logged_loss = avg
            log_loss_sum = 0.0
            log_count = 0
            row = {
                "step": step,
                "loss": avg,
                "lr": lr,
                "seconds": elapsed_offset + (time.time() - t0),
            }
            append_jsonl(log_path, row)
            pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr:.2e}")

        if val_loader is not None and val_every > 0 and step % val_every == 0 and is_main:
            last_val = eval_val_loss(model, val_loader, device, use_bf16)
            if best_val is None or last_val < best_val:
                best_val = last_val
            append_jsonl(
                log_path,
                {
                    "step": step,
                    "val_loss": last_val,
                    "best_val_loss": best_val,
                    "seconds": elapsed_offset + (time.time() - t0),
                },
            )
            pbar.set_postfix(loss=f"{last_logged_loss or 0:.4f}", val=f"{last_val:.4f}")

        need_ckpt = False
        if save_every > 0 and step % save_every == 0:
            need_ckpt = True
        if _save_request["flag"]:
            need_ckpt = True

        if need_ckpt:
            if ddp:
                dist.barrier()
            if is_main:
                save_training_checkpoint(
                    model=model,
                    opt=opt,
                    tok=tok,
                    out_dir=out_dir,
                    step=step,
                    max_steps=max_steps,
                    last_logged_loss=last_logged_loss,
                    best_val=best_val,
                    last_val=last_val,
                    t0=t0 - elapsed_offset,
                    cfg=cfg,
                    pre_save=pre_save,
                    tag="signal" if _save_request["flag"] else "periodic",
                )
            if ddp:
                dist.barrier()
            if _save_request["flag"]:
                if is_main:
                    print("[ckpt] exiting after signal checkpoint", flush=True)
                break

    pbar.close()

    if ddp:
        dist.barrier()

    if is_main:
        outp = ensure_dir(out_dir)
        raw = _raw_model(model)
        # Final resume snapshot first (always)
        save_training_checkpoint(
            model=model,
            opt=opt,
            tok=tok,
            out_dir=out_dir,
            step=step,
            max_steps=max_steps,
            last_logged_loss=last_logged_loss,
            best_val=best_val,
            last_val=last_val,
            t0=t0 - elapsed_offset,
            cfg=cfg,
            pre_save=pre_save,
            tag="final_resume",
        )
        if pre_save is not None:
            pre_save(raw)
        raw.cpu()
        raw.save_pretrained(outp, safe_serialization=True)
        tok.save_pretrained(outp)
        meta = {
            "max_steps": max_steps,
            "completed_steps": step,
            "finished": step >= max_steps,
            "final_loss_proxy": last_logged_loss,
            "final_val_loss": last_val,
            "best_val_loss": best_val,
            "seconds": elapsed_offset + (time.time() - t0),
            "device": str(device),
            "world_size": world_size,
            "target_tokens": tcfg.get("target_tokens"),
            "seed": cfg["seed"],
        }
        with open(Path(outp) / "train_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[train] Saved to {outp} meta={meta}", flush=True)
        _ddp_cleanup(ddp)
        return meta

    _ddp_cleanup(ddp)
    return {"rank": rank, "max_steps": max_steps, "completed_steps": step}
