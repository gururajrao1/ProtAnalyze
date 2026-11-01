"""Train the windowed-linear SS3 head on frozen ESM-2-8M representations.

Reads the JSONL labels from scripts/fetch_ss_data.py, embeds every chain
once with the repo's ESM-2 path (CPU, single forward pass), then trains a
multinomial logistic probe over edge-replicated windows of per-residue
representations. Feature standardisation is folded into the exported
weights, so inference stays a pure-numpy ``feats @ W + b``.

Outputs (all deterministic given the same labels file):
  * data/ss_head_8m.json ........... the shipped head bundle (KBs)
  * data/ss_train_report.json ...... Q3 numbers, sizes, provenance
  * tests/data/ss_pinned20.json .... 20 held-out chains (seq + labels)
  * tests/data/ss_pinned20_reps.npz  matching ESM reps (fp16, hermetic test)

Usage:
  python scripts/train_ss_head.py --labels data/ss_labels.jsonl
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from protein_annotation.embeddings import _try_esm_forward  # noqa: E402
from protein_annotation.structure import (  # noqa: E402
    SS3_LABELS,
    SSHeadWeights,
    predict_ss3_head,
    ss_accuracy,
)

LABEL_INDEX = {s: k for k, s in enumerate(SS3_LABELS)}


def load_label_rows(path: Path) -> list[dict]:
    """Read label JSONL, skipping the provenance first line."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            if "provenance" in obj:
                continue
            assert len(obj["seq"]) == len(obj["ss3"]), obj["id"]
            assert set(obj["ss3"]) <= set(SS3_LABELS), obj["id"]
            rows.append(obj)
    return rows


def split_rows(rows: list[dict], test_every: int = 8, n_pinned: int = 20
               ) -> tuple[list[dict], list[dict], list[dict]]:
    """Deterministic split: sorted by id, every Nth chain held out.

    Returns (train, pinned20, test_rest): the pinned slice is the first
    `n_pinned` held-out chains and ships as the pytest fixture.
    """
    ordered = sorted(rows, key=lambda r: r["id"])
    test = [r for k, r in enumerate(ordered) if k % test_every == 0]
    train = [r for k, r in enumerate(ordered) if k % test_every != 0]
    if len(test) < n_pinned:
        raise ValueError(f"only {len(test)} held-out chains (< {n_pinned})")
    return train, test[:n_pinned], test[n_pinned:]


def windowed_features(reps: list[np.ndarray], window: int) -> tuple[np.ndarray, np.ndarray]:
    """Stack edge-replicated window features + owner index per residue."""
    feats, owners = [], []
    span = 2 * window + 1
    for k, rep in enumerate(reps):
        padded = np.pad(np.asarray(rep, dtype=np.float64), ((window, window), (0, 0)), mode="edge")
        n = rep.shape[0]
        feats.append(np.concatenate([padded[i:i + n] for i in range(span)], axis=1))
        owners.append(np.full(n, k, dtype=np.int64))
    return np.concatenate(feats, axis=0), np.concatenate(owners)


def fit_probe_torch(X: np.ndarray, y: np.ndarray, epochs: int, lr: float,
                    batch_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Minibatch-Adam multinomial logistic regression (CPU, deterministic)."""
    import torch

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    n, d = X.shape
    counts = np.bincount(y, minlength=3).astype(np.float64)
    class_w = torch.as_tensor(n / (3.0 * np.maximum(counts, 1)), dtype=torch.float32)
    model = torch.nn.Linear(d, 3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_w)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    with torch.no_grad():
        W = model.weight.detach().cpu().numpy().T  # (d, 3)
        b = model.bias.detach().cpu().numpy()
    return W, b


def main(argv: list[str] | None = None) -> int:
    """CLI: train head, evaluate Q3, write bundle + fixtures."""
    parser = argparse.ArgumentParser(description="Train the SS3 head on frozen ESM-2-8M.")
    parser.add_argument("--labels", default="data/ss_labels.jsonl", help="fetch_ss_data.py JSONL.")
    parser.add_argument("--head-out", default="data/ss_head_8m.json", help="Head bundle path.")
    parser.add_argument("--report-out", default="data/ss_train_report.json", help="Report path.")
    parser.add_argument("--window", type=int, default=7, help="One-sided context width.")
    parser.add_argument("--epochs", type=int, default=30, help="Adam epochs.")
    parser.add_argument("--lr", type=float, default=3e-3, help="Adam learning rate.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Minibatch rows.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed.")
    parser.add_argument("--esm-model", default="esm2_t6_8M_UR50D", help="ESM checkpoint.")
    args = parser.parse_args(argv)

    rows = load_label_rows(Path(args.labels))
    train_rows, pinned, test_rest = split_rows(rows)
    print(f"train={len(train_rows)} pinned={len(pinned)} test_rest={len(test_rest)}", flush=True)

    # One shared ESM forward pass over every chain (train + held-out).
    all_rows = train_rows + pinned + test_rest
    forwarded = _try_esm_forward([r["seq"] for r in all_rows], args.esm_model)
    if forwarded is None:
        print("error: ESM-2 forward failed (need torch + fair-esm + weights)", file=sys.stderr)
        return 1
    reps, clipped = forwarded
    for rep, row, clip in zip(reps, all_rows, clipped):
        if len(clip) != len(row["seq"]) or rep.shape[0] != len(row["seq"]):
            print(f"error: ESM clipped {row['id']} ({len(row['seq'])} residues)", file=sys.stderr)
            return 1
    dim = int(reps[0].shape[1])
    n_train = len(train_rows)

    X_all, _ = windowed_features(reps, args.window)
    y_all = np.array([LABEL_INDEX[s] for r in all_rows for s in r["ss3"]], dtype=np.int64)
    train_len = sum(len(r["seq"]) for r in train_rows)
    X_train, y_train = X_all[:train_len], y_all[:train_len]

    # Standardise on train only, then fold mean/std into the weights so
    # inference needs no scaler: (x-mu)/sd @ W + b == x @ (W/sd) + (b - mu/sd @ W).
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd < 1e-6] = 1.0
    W, b = fit_probe_torch((X_train - mu) / sd, y_train, args.epochs, args.lr,
                           args.batch_size, args.seed)
    W = W / sd[:, None]
    b = b - (mu / sd) @ W

    head = SSHeadWeights(dim=dim, window=args.window, labels=list(SS3_LABELS),
                         weights=W.tolist(), bias=b.tolist(),
                         esm_model=args.esm_model,
                         trained_on=f"rcsb-pdb {len(train_rows)} chains, {_dt.date.today().isoformat()}")
    head.save(ROOT / args.head_out)

    def q3(sub_rows: list[dict], sub_reps: list[np.ndarray]) -> float:
        pred = [s for rep in sub_reps for s in predict_ss3_head(rep, head)]
        true = [s for r in sub_rows for s in r["ss3"]]
        return ss_accuracy(pred, true)

    train_q3 = q3(train_rows, reps[:n_train])
    pinned_q3 = q3(pinned, reps[n_train:n_train + len(pinned)])
    rest_q3 = q3(test_rest, reps[n_train + len(pinned):]) if test_rest else 0.0
    print(f"Q3 train={train_q3:.4f} pinned20={pinned_q3:.4f} test_rest={rest_q3:.4f}", flush=True)

    report = {"date": _dt.date.today().isoformat(), "esm_model": args.esm_model,
              "window": args.window, "epochs": args.epochs, "lr": args.lr,
              "seed": args.seed, "dim": dim,
              "n_train": len(train_rows), "n_pinned": len(pinned), "n_test_rest": len(test_rest),
              "train_residues": int(train_len), "train_q3": train_q3,
              "pinned_q3": pinned_q3, "test_rest_q3": rest_q3,
              "pinned_ids": [r["id"] for r in pinned]}
    (ROOT / args.report_out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Hermetic pytest fixtures: pinned seqs+labels plus fp16 ESM reps.
    tests_data = ROOT / "tests" / "data"
    tests_data.mkdir(parents=True, exist_ok=True)
    (tests_data / "ss_pinned20.json").write_text(
        json.dumps([{"id": r["id"], "seq": r["seq"], "ss3": r["ss3"]} for r in pinned]), encoding="utf-8")
    pinned_reps = reps[n_train:n_train + len(pinned)]
    np.savez_compressed(tests_data / "ss_pinned20_reps.npz",
                        **{f"rep_{k}": np.asarray(r, dtype=np.float16) for k, r in enumerate(pinned_reps)})
    print(f"head -> {args.head_out}; fixtures -> tests/data/ ({pinned_reps[0].nbytes} B/rep-0 fp16)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
