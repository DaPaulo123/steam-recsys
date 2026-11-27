"""RQ1: giờ chơi làm trọng số tin cậy có giúp không? NGƯỜI B — ngày 11-14.

16 lần chạy = 4 hàm confidence x 4 giá trị alpha. Mọi thứ khác cố định.

Quy trình (PLANB mục 9), tách HAI giai đoạn để không lỡ tay chạm vào test:

    # 1-2. Quét trên valid, chọn cấu hình thắng -> results/rq1_best.json
    python -m src.experiments.rq1 --stage sweep --config config/default.yaml --data data/splits

    #      Nếu alpha thắng nằm ở BIÊN dải: mở rộng rồi quét tiếp (không sửa config chung)
    python -m src.experiments.rq1 --stage sweep --alphas 100,200,400

    # 3-4. Học lại trên train+valid, chấm test MỘT LẦN DUY NHẤT
    python -m src.experiments.rq1 --stage test --config config/default.yaml --data data/splits
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# TODO khi A xong src/data/signals.py: đổi dòng dưới thành
#     from src.data.signals import build_confidence
# rồi xóa src/experiments/confidence_stub.py. Đây là chỗ DUY NHẤT phải sửa.
from src.experiments.confidence_stub import build_confidence
from src.eval.evaluate import log_run, run_eval
from src.models.wmf import WMF

RQ1_TABLE = "rq1_runs.csv"        # bảng riêng của RQ1 (C vẽ đường nDCG theo alpha)
BEST_FILE = "rq1_best.json"       # cấu hình thắng trên valid
TEST_DONE = "rq1_test_done.json"  # đánh dấu đã chấm test -> không cho chạy lại
METRIC = "ndcg@10"


# --------------------------------------------------------------------------
# Tiện ích
# --------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_maps(data_dir: str) -> dict:
    """maps.json nằm ở <thư mục cha của splits>/interim/maps.json
    (đúng cho cả data/splits lẫn data_fake/splits)."""
    path = Path(data_dir).parent / "interim" / "maps.json"
    return json.loads(path.read_text(encoding="utf-8"))


def results_dir(cfg: dict) -> Path:
    d = Path(cfg["paths"]["results"]).parent
    d.mkdir(parents=True, exist_ok=True)
    return d


def snapshot_config(config_path: str, cfg: dict) -> str:
    """Chép config vào results/ (config/default.yaml dòng 1 yêu cầu). Trả về tên file."""
    name = f"config_rq1_{datetime.now():%Y%m%d_%H%M%S}.yaml"
    shutil.copyfile(config_path, results_dir(cfg) / name)
    return name


def _hparams(cfg: dict) -> dict:
    w = cfg["wmf"]
    return {
        "factors": w["factors"],
        "regularization": w["regularization"],
        "iterations": w["iterations"],
        "seed": cfg["seed"],
        "cap_hours": cfg["rq1"]["cap_hours"],
    }


def _train_wmf(train_df, scheme: str, alpha: float, cfg: dict, maps: dict) -> tuple[WMF, float]:
    hp = _hparams(cfg)
    conf = build_confidence(train_df, scheme=scheme, alpha=alpha, cap_hours=hp["cap_hours"])
    model = WMF(
        maps["n_users"], maps["n_items"],
        factors=hp["factors"], regularization=hp["regularization"],
        iterations=hp["iterations"], seed=hp["seed"],
    )
    t = time.time()
    model.fit(train_df, confidence=conf)
    return model, time.time() - t


# --------------------------------------------------------------------------
# Các bước của khung
# --------------------------------------------------------------------------

def run_one(train_df, valid_df, scheme: str, alpha: float, cfg: dict, maps: dict) -> dict:
    """Một lần chạy: dựng confidence, huấn luyện WMF, chấm trên tập kiểm định."""
    model, fit_s = _train_wmf(train_df, scheme, alpha, cfg, maps)
    metrics = run_eval(model, train_df, valid_df, k_list=tuple(cfg["eval"]["k_list"]))
    return {
        "split": "valid",
        "scheme": scheme,
        "alpha": float(alpha),
        **_hparams(cfg),
        "fit_seconds": round(fit_s, 2),
        **{k: float(v) for k, v in metrics.items()},
    }


def sweep(train_df, valid_df, cfg: dict, maps: dict, alphas, config_file: str) -> pd.DataFrame:
    """Quét toàn bộ lưới và ghi vào results/runs.csv.

    Ghi dần sau MỖI lần chạy, đừng đợi hết 16 lần. Máy sập giữa chừng thì không
    mất hết. Ghi hai nơi: runs.csv (nhật ký chung, qua log_run của C) và
    rq1_runs.csv (bảng của RQ1).
    """
    table = results_dir(cfg) / RQ1_TABLE
    rows = []
    grid = [(s, float(a)) for s in cfg["rq1"]["schemes"] for a in alphas]
    for n, (scheme, alpha) in enumerate(grid, 1):
        row = run_one(train_df, valid_df, scheme, alpha, cfg, maps)
        row["config_file"] = config_file
        row["timestamp"] = datetime.now().isoformat(timespec="seconds")
        metrics = {k: row[k] for k in row if "@" in k or k == "mrr"}
        log_run(metrics, {k: v for k, v in row.items() if k not in metrics},
                path=cfg["paths"]["results"])
        pd.DataFrame([row]).to_csv(table, mode="a", header=not table.exists(), index=False)
        rows.append(row)
        print(f"[{n:>2}/{len(grid)}] {scheme:<7} alpha={alpha:<6g} "
              f"{METRIC}={row[METRIC]:.4f}  fit {row['fit_seconds']}s")
    return pd.DataFrame(rows)


def pick_best(runs: pd.DataFrame) -> dict:
    """Chọn cấu hình tốt nhất theo nDCG@10 TRÊN TẬP KIỂM ĐỊNH.

    Với mỗi (scheme, alpha) chỉ giữ lần chạy mới nhất.

    Kiểm tra biên cho TỪNG scheme, không chỉ scheme thắng: RQ1 so sánh các hàm
    confidence với nhau, nên nếu đỉnh của một scheme nằm ở biên dải thì đỉnh thật
    của nó chưa được tìm thấy và phép so sánh chưa công bằng
    (src/experiments/README.md: đỉnh ở biên -> mở rộng dải).
    """
    runs = runs[runs["split"] == "valid"].drop_duplicates(["scheme", "alpha"], keep="last")
    best = runs.loc[runs[METRIC].idxmax()]
    edges = {}
    for scheme, g in runs.groupby("scheme"):
        tried = sorted(g["alpha"])
        peak = g.loc[g[METRIC].idxmax(), "alpha"]
        if len(tried) > 1 and peak == tried[0]:
            edges[scheme] = {"side": "low", "peak_alpha": float(peak), "alpha_tried": [float(a) for a in tried]}
        elif peak == tried[-1]:
            edges[scheme] = {"side": "high", "peak_alpha": float(peak), "alpha_tried": [float(a) for a in tried]}
    return {
        "scheme": best["scheme"],
        "alpha": float(best["alpha"]),
        METRIC: float(best[METRIC]),
        "edges": edges,  # {} nghĩa là mọi scheme đều có đỉnh nằm TRONG dải
        "n_runs": int(len(runs)),
    }


def final_test(train_df, test_df, best: dict, cfg: dict, maps: dict) -> dict:
    """Chạy MỘT LẦN DUY NHẤT trên tập kiểm thử với cấu hình đã chọn.

    Đây là con số vào báo cáo. Không chạy lại, không tinh chỉnh thêm sau bước này.

    train_df ở đây là TRAIN + VALID gộp lại (PLANB mục 9 bước 3): confidence tính
    lại trên tập gộp, và tập gộp cũng là train_df của run_eval để exclude_seen,
    item_pop, n_users đều tính trên đúng dữ liệu mô hình đã thấy.
    """
    model, fit_s = _train_wmf(train_df, best["scheme"], best["alpha"], cfg, maps)
    metrics = run_eval(model, train_df, test_df, k_list=tuple(cfg["eval"]["k_list"]))
    return {
        "split": "test",
        "scheme": best["scheme"],
        "alpha": best["alpha"],
        **_hparams(cfg),
        "fit_seconds": round(fit_s, 2),
        **{k: float(v) for k, v in metrics.items()},
    }


# --------------------------------------------------------------------------
# Điểm vào
# --------------------------------------------------------------------------

def main(config_path: str, data_dir: str, stage: str, alphas: str | None = None) -> None:
    cfg = load_config(config_path)
    maps = load_maps(data_dir)
    rdir = results_dir(cfg)
    splits = Path(data_dir)
    train = pd.read_parquet(splits / "train.parquet")
    valid = pd.read_parquet(splits / "valid.parquet")

    if stage == "sweep":
        grid = [float(a) for a in alphas.split(",")] if alphas else cfg["rq1"]["alphas"]
        config_file = snapshot_config(config_path, cfg)
        sweep(train, valid, cfg, maps, grid, config_file)
        best = pick_best(pd.read_csv(rdir / RQ1_TABLE))  # đọc cả các lần quét trước
        (rdir / BEST_FILE).write_text(json.dumps(best, indent=2), encoding="utf-8")
        print(f"\nThắng trên valid: {best['scheme']} alpha={best['alpha']:g} "
              f"{METRIC}={best[METRIC]:.4f}  ({best['n_runs']} cấu hình) -> {BEST_FILE}")
        for scheme, e in best["edges"].items():
            side = "nhỏ nhất" if e["side"] == "low" else "lớn nhất"
            print(f"⚠ {scheme}: đỉnh ở alpha={e['peak_alpha']:g}, giá trị {side} đã thử "
                  f"{e['alpha_tried']} -> MỞ RỘNG dải (--alphas) trước khi chấm test.")
        return

    if stage == "test":
        done = rdir / TEST_DONE
        if done.exists():
            raise SystemExit(f"Đã chấm test rồi ({done}). Test chỉ được chạy MỘT LẦN — "
                             "xóa file này chỉ khi cả nhóm đồng ý chạy lại.")
        best_path = rdir / BEST_FILE
        if not best_path.exists():
            raise SystemExit("Chưa có rq1_best.json — chạy --stage sweep trước.")
        best = json.loads(best_path.read_text(encoding="utf-8"))
        if best["edges"]:
            raise SystemExit(f"Còn scheme có đỉnh ở biên dải alpha: {sorted(best['edges'])}. "
                             "Mở rộng dải (--alphas) trước khi chấm test.")
        test = pd.read_parquet(splits / "test.parquet")
        train_valid = pd.concat([train, valid], ignore_index=True)
        config_file = snapshot_config(config_path, cfg)
        row = final_test(train_valid, test, best, cfg, maps)
        row["config_file"] = config_file
        row["timestamp"] = datetime.now().isoformat(timespec="seconds")
        metrics = {k: row[k] for k in row if "@" in k or k == "mrr"}
        log_run(metrics, {k: v for k, v in row.items() if k not in metrics},
                path=cfg["paths"]["results"])
        table = rdir / RQ1_TABLE
        pd.DataFrame([row]).to_csv(table, mode="a", header=not table.exists(), index=False)
        done.write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(f"TEST (một lần): {best['scheme']} alpha={best['alpha']:g} -> "
              + "  ".join(f"{k}={row[k]:.4f}" for k in metrics))
        return

    raise SystemExit(f"--stage phải là 'sweep' hoặc 'test', nhận {stage!r}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--data", default="data/splits")
    p.add_argument("--stage", choices=["sweep", "test"], required=True)
    p.add_argument("--alphas", default=None,
                   help="ghi đè dải alpha, vd. 100,200,400 (không sửa config chung)")
    a = p.parse_args()
    main(a.config, a.data, a.stage, a.alphas)
