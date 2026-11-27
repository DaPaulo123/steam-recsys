"""Lớp cơ sở cho mọi mô hình. NGƯỜI B — ngày 1-2.

Mọi mô hình trong thư mục này, và cả ContentBased của A, đều tuân thủ giao diện
Recommender trong contracts.py. C chỉ gọi fit và recommend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class BaseRecommender:
    """Xử lý phần chung mà mô hình nào cũng cần.

    Lớp con cài đặt _score_users và đặt lại `name`.
    """

    name = "Base"

    def __init__(self, n_users: int, n_items: int):
        self.n_users = n_users
        self.n_items = n_items
        self._seen: dict[int, set[int]] | None = None
        # Độ phổ biến mỗi game đếm TỪ TẬP HUẤN LUYỆN, làm phương án dự phòng
        # cho người dùng không có trong train (contracts.py: Recommender.recommend).
        self._pop: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame, confidence: np.ndarray | None = None) -> None:
        """Lưu lịch sử để exclude_seen hoạt động, đếm độ phổ biến cho phương án
        dự phòng, rồi gọi _fit của lớp con.
        """
        self._seen = train_df.groupby("u_idx")["i_idx"].apply(set).to_dict()
        self._pop = np.bincount(
            train_df["i_idx"].to_numpy(), minlength=self.n_items
        ).astype(np.float32)
        self._fit(train_df, confidence)

    def _fit(self, train_df: pd.DataFrame, confidence: np.ndarray | None) -> None:
        raise NotImplementedError("lớp con cài đặt")

    def _score_users(self, user_ids: np.ndarray) -> np.ndarray:
        """Trả về mảng điểm (len(user_ids), n_items). Lớp con cài đặt."""
        raise NotImplementedError("lớp con cài đặt")

    def recommend(self, user_ids, k: int = 10, exclude_seen: bool = True) -> np.ndarray:
        """Chấm điểm, dự phòng cho người dùng mới, loại game đã xem, lấy top-k.

        Trả về int32 kích thước (len(user_ids), k), không có -1 hay NaN.
        """
        if self._seen is None or self._pop is None:
            raise RuntimeError(f"{self.name}: phải gọi fit trước recommend")
        k = int(k)
        if not 1 <= k <= self.n_items:
            raise ValueError(f"k phải trong [1, {self.n_items}], nhận {k}")

        user_ids = np.asarray(user_ids, dtype=np.int64).ravel()
        n = len(user_ids)

        # np.array luôn SAO CHÉP: lớp con có thể trả về mảng dùng chung
        # (vd. MostPopular lặp một vector), sửa tại chỗ sẽ làm hỏng mô hình.
        scores = np.array(self._score_users(user_ids), dtype=np.float32)
        if scores.shape != (n, self.n_items):
            raise ValueError(
                f"{self.name}._score_users trả về shape {scores.shape}, "
                f"cần {(n, self.n_items)}"
            )
        # argpartition coi NaN là lớn nhất -> game NaN lọt lên top mà không ai biết.
        if np.isnan(scores).any():
            raise ValueError(f"{self.name}._score_users trả về NaN")

        # Dự phòng: người dùng không có trong train nhận điểm theo độ phổ biến.
        cold = np.fromiter((int(u) not in self._seen for u in user_ids), dtype=bool, count=n)
        if cold.any():
            scores[cold] = self._pop

        if exclude_seen:
            for r, u in enumerate(user_ids):
                seen = self._seen.get(int(u))
                if seen:
                    scores[r, list(seen)] = -np.inf

        # Lấy k ứng viên điểm cao nhất (không cần sắp toàn bộ catalog),
        # rồi chỉ sắp k ứng viên đó theo điểm giảm dần.
        part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1, kind="stable")
        top = np.take_along_axis(part, order, axis=1)
        return top.astype(np.int32)


# --------------------------------------------------------------------------
# Dùng chung khi xuất / nạp file cho demo (knn.py, wmf.py)
# --------------------------------------------------------------------------

def _app_ids_in_order(items: pd.DataFrame, n_items: int) -> np.ndarray:
    """app_id sắp theo i_idx. Báo lỗi nếu i_idx không phủ đúng 0..n_items-1."""
    idx = items["i_idx"].to_numpy()
    if len(idx) != n_items or not np.array_equal(np.sort(idx), np.arange(n_items)):
        raise ValueError(
            f"items có {len(idx)} dòng / i_idx không phủ đúng 0..{n_items - 1}: "
            "items.parquet không khớp với mô hình"
        )
    return items.sort_values("i_idx")["app_id"].to_numpy(dtype=np.int64)


def _check_saved_app_ids(saved, items: pd.DataFrame, n_items: int, fname: str, what: str) -> None:
    """So app_id đã lưu lúc huấn luyện với items.parquet hiện tại.

    Lệch nghĩa là A đã chạy lại preprocess -> i_idx trỏ tới game khác -> B phải
    xuất lại ``what`` (xem PLANB.md).
    """
    saved = np.asarray(saved, dtype=np.int64)
    try:
        current = _app_ids_in_order(items, n_items)
    except ValueError as e:
        raise ValueError(f"{fname}: {e}. B cần xuất lại {what}.") from e
    if not np.array_equal(saved, current):
        n_diff = int((saved != current).sum())
        raise ValueError(
            f"{fname}: {n_diff} i_idx trỏ tới game khác so với lúc huấn luyện "
            f"(items.parquet đã bị sinh lại). B cần xuất lại {what}."
        )
