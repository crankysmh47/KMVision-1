"""Checkpoint helpers for Phase C training (ordered step dirs + resume state)."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

import torch

LATEST_JSON = "latest.json"
TRAINING_STATE_FILE = "training_state.pt"
LATEST_TRAINING_STATE = "latest_training_state.pt"
STEP_DIR_RE = re.compile(r"^step_(\d+)$")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def step_dir_for(output_dir: str, global_step: int) -> str:
    return os.path.join(output_dir, f"step_{global_step:06d}")


def parse_step_dir_name(name: str) -> int | None:
    match = STEP_DIR_RE.match(name)
    if not match:
        return None
    return int(match.group(1))


def verify_step_dir(step_dir: str) -> list[str]:
    errors = []
    if not os.path.isdir(step_dir):
        return [f"missing directory: {step_dir}"]
    for name in ("adapter_model.safetensors", "projector_weights.pth", "adapter_config.json"):
        path = os.path.join(step_dir, name)
        if not os.path.isfile(path):
            errors.append(f"missing file: {path}")
    return errors


def find_latest_step_dir(output_dir: str) -> tuple[int | None, str | None]:
    if not os.path.isdir(output_dir):
        return None, None
    best_step: int | None = None
    best_dir: str | None = None
    for name in os.listdir(output_dir):
        step_num = parse_step_dir_name(name)
        if step_num is None:
            continue
        candidate = os.path.join(output_dir, name)
        if not verify_step_dir(candidate):
            if best_step is None or step_num > best_step:
                best_step = step_num
                best_dir = candidate
    return best_step, best_dir


def load_latest_pointer(output_dir: str) -> dict | None:
    path = os.path.join(output_dir, LATEST_JSON)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_latest_pointer(
    output_dir: str,
    *,
    global_step: int,
    step_dir: str,
    micro_step: int,
    max_global_steps: int,
    has_training_state: bool,
    init_checkpoint: str | None = None,
    weights_step_dir: str | None = None,
) -> None:
    weights_dir = (weights_step_dir or step_dir).replace("\\", "/")
    payload = {
        "global_step": global_step,
        "micro_step": micro_step,
        "max_global_steps": max_global_steps,
        "step_dir": step_dir.replace("\\", "/"),
        "weights_step_dir": weights_dir,
        "has_training_state": has_training_state,
        "saved_at_utc": utc_now(),
    }
    if init_checkpoint:
        payload["init_checkpoint"] = init_checkpoint.replace("\\", "/")
    with open(os.path.join(output_dir, LATEST_JSON), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def latest_training_state_path(output_dir: str) -> str:
    return os.path.join(output_dir, LATEST_TRAINING_STATE)


def save_progress(
    output_dir: str,
    *,
    global_step: int,
    micro_step: int,
    weights_step_dir: str,
    max_global_steps: int,
    optimizer,
    scheduler,
    last_loss: float,
    args_dict: dict[str, Any],
    init_checkpoint: str | None = None,
) -> None:
    """Lightweight per-step progress (survives reboot between full checkpoints)."""
    state_path = latest_training_state_path(output_dir)
    save_training_state(
        state_path,
        global_step=global_step,
        micro_step=micro_step,
        optimizer=optimizer,
        scheduler=scheduler,
        last_loss=last_loss,
        args_dict=args_dict,
    )
    save_latest_pointer(
        output_dir,
        global_step=global_step,
        step_dir=weights_step_dir,
        micro_step=micro_step,
        max_global_steps=max_global_steps,
        has_training_state=True,
        init_checkpoint=init_checkpoint,
        weights_step_dir=weights_step_dir,
    )


def save_training_state(
    path: str,
    *,
    global_step: int,
    micro_step: int,
    optimizer,
    scheduler,
    last_loss: float,
    args_dict: dict[str, Any],
) -> None:
    state = {
        "global_step": global_step,
        "micro_step": micro_step,
        "last_loss": last_loss,
        "args": args_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_python": __import__("random").getstate(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, path)


def load_training_state(path: str, *, optimizer, scheduler, device) -> dict:
    # Load on CPU: RNG states must stay ByteTensors on CPU; optimizer loads onto param devices.
    state = torch.load(path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    import random

    if "rng_python" in state:
        random.setstate(state["rng_python"])
    if "rng_torch" in state:
        rng = state["rng_torch"]
        if isinstance(rng, torch.Tensor) and rng.device.type != "cpu":
            rng = rng.cpu()
        torch.set_rng_state(rng)
    if state.get("rng_cuda") is not None and torch.cuda.is_available():
        cuda_rng = state["rng_cuda"]
        if isinstance(cuda_rng, list):
            cuda_rng = [
                t.cpu() if isinstance(t, torch.Tensor) and t.device.type != "cpu" else t
                for t in cuda_rng
            ]
        torch.cuda.set_rng_state_all(cuda_rng)
    return state


def resolve_resume(output_dir: str, init_checkpoint: str | None) -> dict | None:
    """
    Decide how to resume a run.

    Returns None if no resume (fresh run from init_checkpoint).
    Otherwise dict with keys: step_dir, global_step, micro_step, has_training_state, training_state_path.
    """
    pointer = load_latest_pointer(output_dir)
    root_state = latest_training_state_path(output_dir)
    has_root_state = os.path.isfile(root_state)

    if pointer:
        weights_dir = pointer.get("weights_step_dir") or pointer.get("step_dir", "")
        if weights_dir and os.path.isdir(weights_dir) and not verify_step_dir(weights_dir):
            step_dir = weights_dir
            training_state_path = root_state if has_root_state else os.path.join(
                step_dir, TRAINING_STATE_FILE
            )
            return {
                "step_dir": step_dir,
                "global_step": int(pointer.get("global_step", 0)),
                "micro_step": int(pointer.get("micro_step", 0)),
                "max_global_steps": int(pointer.get("max_global_steps", 2000)),
                "has_training_state": has_root_state or os.path.isfile(training_state_path),
                "training_state_path": training_state_path if (has_root_state or os.path.isfile(training_state_path)) else root_state,
            }

    step_num, step_dir = find_latest_step_dir(output_dir)
    if step_dir is None:
        return None

    training_state_path = os.path.join(step_dir, TRAINING_STATE_FILE)
    global_step = step_num or 0
    micro_step = 0
    meta_path = os.path.join(step_dir, "checkpoint_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        global_step = int(meta.get("global_step", global_step))
        micro_step = int(meta.get("micro_step", 0))

    use_root = has_root_state
    return {
        "step_dir": step_dir,
        "global_step": global_step,
        "micro_step": micro_step,
        "max_global_steps": 2000,
        "has_training_state": use_root or os.path.isfile(training_state_path),
        "training_state_path": root_state if use_root else training_state_path,
    }
