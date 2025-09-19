import json, math, os, random, time, argparse, re
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader

from torch.cuda.amp import GradScaler, autocast

INPUT = 776
PHASE_META_IDX = 12 * 64 + 6  # 774
DEFAULT_CP_SCALE = 1000.0

def load_weights_json_into_model(model: nn.Module, json_path):
    """Load weights exported by export_weights_json(...) back into the PyTorch model."""
    with open(json_path, "r") as f:
        layers = json.load(f)  # list of {"W": [[out][in]], "b": [out]}

    # Collect linear layers in order
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(linear_layers) != len(layers):
        raise ValueError(f"Layer count mismatch: model has {len(linear_layers)} Linear layers, file has {len(layers)}")

    for (lin, blob) in zip(linear_layers, layers):
        W = torch.tensor(blob["W"], dtype=lin.weight.dtype)
        b = torch.tensor(blob["b"], dtype=lin.bias.dtype)
        if W.shape != lin.weight.data.shape or b.shape != lin.bias.data.shape:
            raise ValueError(f"Shape mismatch: got W {W.shape}, b {b.shape} but layer expects {lin.weight.data.shape}, {lin.bias.data.shape}")
        lin.weight.data.copy_(W)   # file is [out, in], matches nn.Linear
        lin.bias.data.copy_(b)

class EvalNet(nn.Module):
    def __init__(self, h1=256, h2=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT, h1), nn.ReLU(inplace=True),
            nn.Linear(h1, h2), nn.ReLU(inplace=True),
            nn.Linear(h2, 2),
        )
    def forward(self, x): return self.net(x)

class JSONLShardStreamer(IterableDataset):
    def __init__(self, files: List[str], cp_scale: float = DEFAULT_CP_SCALE, buffer_size: int = 4096):
        super().__init__()
        self.files = files; self.cp_scale = float(cp_scale); self.buffer_size = int(buffer_size)
    def _norm_cp(self, v: float) -> float:
        v = max(-self.cp_scale, min(self.cp_scale, float(v))); return v / self.cp_scale
    def parse_record(self, rec: Dict):
        if "x" not in rec: return None
        x = rec["x"]
        if not isinstance(x, list) or len(x) != INPUT: return None
        x_t = torch.tensor(x, dtype=torch.float32)
        phase = float(x[PHASE_META_IDX]) if 0 <= PHASE_META_IDX < len(x) else 0.0
        phase = 0.0 if math.isnan(phase) else max(0.0, min(1.0, phase))
        label: Dict[str, float] = {"phase": phase}
        if "mg_norm" in rec and "eg_norm" in rec:
            label["mg_norm"] = float(rec["mg_norm"]); label["eg_norm"] = float(rec["eg_norm"]); return x_t, label
        if "mg" in rec and "eg" in rec:
            label["mg_norm"] = self._norm_cp(rec["mg"]); label["eg_norm"] = self._norm_cp(rec["eg"]); return x_t, label
        if "cp" in rec:
            label["cp_norm"] = self._norm_cp(rec["cp"]); return x_t, label
        if "y" in rec:
            y = float(rec["y"]); label["cp_norm"] = max(-1.0, min(1.0, y)); return x_t, label
        return None
    def __iter__(self):
        rng = random.Random(os.getpid() ^ int(time.time()))
        files = list(self.files); rng.shuffle(files)
        buf: List[Tuple[torch.Tensor, Dict]] = []
        for path in files:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if not line.strip(): continue
                        try: rec = json.loads(line)
                        except Exception: continue
                        parsed = self.parse_record(rec)
                        if parsed is None: continue
                        buf.append(parsed)
                        if len(buf) >= self.buffer_size:
                            rng.shuffle(buf)
                            while buf: yield buf.pop()
                rng.shuffle(buf)
                while buf: yield buf.pop()
            except FileNotFoundError:
                continue

def export_weights_json(model: nn.Module, out_path: Path):
    layers = []
    sd = model.state_dict()
    names = [k for k in sd.keys() if k.endswith('.weight')]; names.sort()
    for w_name in names:
        b_name = w_name.replace('weight','bias')
        W = sd[w_name].cpu().numpy().tolist()
        b = sd[b_name].cpu().numpy().tolist()
        layers.append({"W": W, "b": b})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(layers, f, separators=(",",":"))

def human_count(n: int) -> str: return f"{n:,}".replace(",","_")

def setup_run_dir(base_dir: Path, run_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / "runs" / f"{ts}-{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def append_jsonl(path: Path, obj: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, separators=(",",":")) + "\n")

def collate_keep_labels(batch):
    xs, ys = [], []
    for x, lab in batch:
        if "cp_norm" in lab:
            y = float(lab["cp_norm"])
        elif "mg_norm" in lab and "eg_norm" in lab:
            phase = float(lab.get("phase", 0.5))
            y = phase * float(lab["mg_norm"]) + (1.0 - phase) * float(lab["eg_norm"])
        elif "y" in lab:
            y = max(-1.0, min(1.0, float(lab["y"])))
        else:
            # skip malformed items
            continue
        xs.append(x)
        ys.append(y)
    return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.float32)

# --------- NEW: resume helpers ---------
def find_latest_run(base_dir: Path, run_name_filter: Optional[str]) -> Optional[Path]:
    runs_dir = base_dir / "runs"
    if not runs_dir.exists(): return None
    candidates = []
    for child in runs_dir.iterdir():
        if not child.is_dir(): continue
        if run_name_filter and not child.name.endswith(run_name_filter): continue
        try:
            # prefix is YYYYmmdd-HHMMSS
            datetime.strptime(child.name.split("-")[0] + "-" + child.name.split("-")[1], "%Y%m%d-%H%M%S")
            candidates.append(child)
        except Exception:
            continue
    if not candidates: return None
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]

def last_checkpoint_epoch(run_dir: Path) -> int:
    mx = 0
    for p in run_dir.glob("checkpoint_epoch_*.pt"):
        m = re.search(r"checkpoint_epoch_(\d+)\.pt$", p.name)
        if m: mx = max(mx, int(m.group(1)))
    return mx

def load_cfg(run_dir: Path) -> Dict:
    cfg_path = run_dir / "cfg.json"
    return json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

# --------------------------------------

def train(
    data_globs: List[str],
    out_dir: str = ".",
    run_name: str = "nnue_live",
    epochs: int = 2,
    batch_size: int = 8192,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    workers: int = 4,
    buffer_size: int = 4096,
    cp_scale: float = DEFAULT_CP_SCALE,
    h1: int = 256,
    h2: int = 64,
    grad_clip: float = 1.0,
    amp: bool = True,
    seed: int = 42,
    resume: Optional[str] = None,
    resume_latest: bool = False,
    fresh: bool = False,
    checkpoint_every_steps: int = 0,
):
    """
    Trains an NNUE-style 776->256->64->2 MG/EG network on streaming JSONL.
    Exports:
      - frequent step checkpoints if checkpoint_every_steps > 0 (atomic latest)
      - end-of-epoch JSON weights + (optional) torch checkpoint
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(seed); torch.manual_seed(seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # new run dir unless resuming latest
    if resume_latest:
        # pick the newest run dir under out_dir/runs
        runs_root = out_dir / "runs"
        runs_root.mkdir(exist_ok=True)
        candidates = list(runs_root.glob("*"))
        run_dir = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else (runs_root / f"{int(time.time())}-{run_name}")
    else:
        run_dir = (out_dir / "runs" / f"{int(time.time())}-{run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save minimal config for reproducibility
    (run_dir/"cfg.json").write_text(json.dumps({
        "data": data_globs, "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "weight_decay": weight_decay, "workers": workers, "buffer_size": buffer_size,
        "cp_scale": cp_scale, "h1": h1, "h2": h2, "amp": amp, "seed": seed,
        "input": int(INPUT), "phase_meta_idx": int(PHASE_META_IDX),
        "checkpoint_every_steps": int(checkpoint_every_steps),
    }, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model / optimizer / scaler ---
    model = EvalNet(h1=h1, h2=h2).to(device)  # adjust ctor if yours differs
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=True if torch.cuda.is_available() else False)
    scaler = GradScaler(enabled=amp)
    loss_fn = nn.MSELoss()

    # --- (Optional) resume from a checkpoint or latest json ---
    start_epoch = 0
    global_step = 0
    if resume and (Path(resume).exists()):
        ckpt = torch.load(resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("step", 0)
    elif resume_latest:
        latest_json = run_dir / "weights_latest.json"
        if latest_json.exists():
            try:
                load_weights_json_into_model(model, latest_json)
                print(f"[init] Loaded JSON weights: {latest_json}")
            except Exception as e:
                print(f"[init] Warning: could not load JSON weights: {e}")

    # --- Data loader (streaming) ---
    files = []
    for g in data_globs:
        files += glob(g)
    ds = JSONLShardStreamer(files, cp_scale=cp_scale, buffer_size=buffer_size)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=workers, 
                    pin_memory=(device.type=='cuda'), 
                    prefetch_factor=2 if workers>0 else None, 
                    collate_fn=collate_keep_labels)

    model.train()
    t0 = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        samples = 0
        files_seen_before = getattr(ds, "files_seen", 0)

        for batch in dl:
            x, y = batch
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                pred2 = model(x)  # (B, 2)
                if pred2.ndim == 2 and pred2.size(-1) == 2:
                    pred = pred2.mean(dim=-1)  # simple MG/EG average; replace with your canonical blend if desired
                else:
                    pred = pred2.squeeze(-1)
                loss = loss_fn(pred, y)

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            # --- frequent checkpointing (atomic latest) ---
            if checkpoint_every_steps and global_step > 0 and (global_step % checkpoint_every_steps == 0):
                tmp = run_dir / f".latest_tmp_{epoch:03d}_{global_step:07d}.json"
                export_weights_json(model, tmp)          # write to temp
                latest = run_dir / "weights_latest.json"
                tmp.replace(latest)                      # atomic rename on same filesystem

            # stats
            bs = x.shape[0]
            epoch_loss += loss.item() * bs
            samples += bs
            global_step += 1

            # (Optional) lightweight prints every ~N steps
            if (global_step % 200 == 0):
                elapsed = time.time() - t0
                mse = epoch_loss / max(1, samples)
                files_seen_now = getattr(ds, "files_seen", 0)
                print(f"[epoch {epoch+1}] step {global_step} samples={samples:_} files={files_seen_now} mse={mse:.6f} time={elapsed:.1f}s")

        # ----- end of epoch: checkpoints -----

        # 2) export a human-readable JSON weight snapshot
        epoch_json = run_dir / f"weights_epoch_{epoch+1:03d}.json"
        export_weights_json(model, epoch_json)

        # 3) refresh weights_latest.json atomically
        latest = run_dir / "weights_latest.json"
        tmp_epoch = run_dir / f".latest_tmp_epoch_{epoch+1:03d}.json"
        tmp_epoch.write_text(epoch_json.read_text())
        tmp_epoch.replace(latest)

        # epoch summary
        mse = epoch_loss / max(1, samples)
        files_seen_now = getattr(ds, "files_seen", 0)
        print(f"[epoch {epoch+1}/{epochs}] MSE={mse:.6f}  samples={samples:_}  files={files_seen_now - files_seen_before}")

    print("Training complete.")

def main():
    ap = argparse.ArgumentParser(description="Fixed NNUE trainer with explicit output directory + resume.")
    ap.add_argument("--data", nargs="+", required=True, help="One or more JSONL paths or globs")
    ap.add_argument("--out-dir", default=".", help="Directory to write runs/… into (e.g., C:/Users/PC/Desktop/nnue_out)")
    ap.add_argument("--run-name", default="nnue")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--buffer-size", type=int, default=4096)
    ap.add_argument("--cp-scale", type=float, default=DEFAULT_CP_SCALE)
    ap.add_argument("--h1", type=int, default=256)
    ap.add_argument("--h2", type=int, default=64)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    # NEW:
    ap.add_argument("--resume", type=str, default=None, help="Path to a specific runs/<ts-name> folder to continue")
    ap.add_argument("--resume-latest", action="store_true", help="Auto-pick newest runs/<ts-name> under --out-dir (filtered by --run-name)")
    ap.add_argument("--fresh", action="store_true", help="Force starting a brand new run (ignore resume flags)")
    ap.add_argument("--checkpoint-every-steps", type=int, default=0,
                help="If >0, also export weights_latest.json every K steps (in addition to end-of-epoch).")
    args = ap.parse_args()
    train(
        data_globs=args.data,
        out_dir=args.out_dir,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        workers=args.workers,
        buffer_size=args.buffer_size,
        cp_scale=args.cp_scale,
        h1=args.h1, h2=args.h2,
        amp=not args.no_amp,
        seed=args.seed,
        resume=args.resume,
        resume_latest=args.resume_latest,
        fresh=args.fresh,
        checkpoint_every_steps = args.checkpoint_every_steps,
    )

if __name__ == "__main__":
    main()