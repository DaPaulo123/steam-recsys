"""ItemKNN — lọc cộng tác láng giềng. NGƯỜI B — ngày 4-5.

Hai game giống nhau nếu hay được cùng một nhóm người chơi.

    sim(i,j) = |U_i giao U_j| / (sqrt|U_i| * sqrt|U_j|)
    s(u,i)   = tổng sim(i,j) với j chạy qua lịch sử của u

Giá trị riêng của mô hình này: nó giải thích được. "Gợi ý Dead Cells vì bạn đã chơi
Hades" — các mô hình nhân tố ẩn không làm được điều đó. Vì vậy demo dùng nó.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from implicit.nearest_neighbours import CosineRecommender
from scipy.sparse import csr_matrix, load_npz, save_npz

from .base import BaseRecommender, _app_ids_in_order, _check_saved_app_ids


class ItemKNN(BaseRecommender):
    name = "ItemKNN"

    def __init__(self, n_users: int, n_items: int, k_neighbors: int = 50):
        super().__init__(n_users, n_items)
        self.k_neighbors = k_neighbors
        self.sim = None  # ma trận tương đồng (n_items, n_items), thưa
        self._X = None   # ma trận tương tác (n_users, n_items), thưa, nhị phân

    def _fit(self, train_df: pd.DataFrame, confidence: np.ndarray | None) -> None:
        """Dùng implicit.nearest_neighbours.CosineRecommender.

        Đã kiểm tra với implicit 0.7.3: CosineRecommender.fit nhận ma trận
        (n_users, n_items) — nó chuẩn hóa từng CỘT (mỗi cột là một game) rồi
        tính tương đồng giữa các cột. Hàng i của .similarity là K láng giềng của i.

        confidence bị bỏ qua: tương đồng tính trên ma trận NHỊ PHÂN để đúng công
        thức |U_i giao U_j| ở đầu file.
        """
        rows = train_df["u_idx"].to_numpy()
        cols = train_df["i_idx"].to_numpy()
        X = csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(self.n_users, self.n_items),
        )
        X.data[:] = 1.0  # phòng trùng cặp (u, i): csr_matrix cộng dồn bản trùng

        model = CosineRecommender(K=self.k_neighbors)
        model.fit(X, show_progress=False)
        self.sim = model.similarity.tocsr()
        if self.sim.shape != (self.n_items, self.n_items):
            raise ValueError(f"similarity có shape {self.sim.shape}, cần ({self.n_items}, {self.n_items})")
        self._X = X

    def _score_users(self, user_ids: np.ndarray) -> np.ndarray:
        """s(u, i) = tổng sim(j, i) với j trong lịch sử của u.

        Người dùng không có lịch sử ra hàng toàn 0; BaseRecommender thay bằng
        độ phổ biến. Loại game đã xem và cắt top-k cũng do BaseRecommender làm.
        """
        return (self._X[user_ids] @ self.sim).toarray()

    def export_similarity(self, path: str, items: pd.DataFrame) -> None:
        """Xuất ma trận tương đồng cho A dùng trong demo. Hạn: ngày 5.

        Ghi HAI file cạnh nhau (đuôi trong ``path`` bị thay):
          <path>.npz   ma trận thưa (scipy.sparse.save_npz)
          <path>.json  {model, k_neighbors, n_items, app_ids}
        app_ids là app_id theo đúng thứ tự i_idx lúc huấn luyện. A nạp bằng
        load_similarity để phát hiện khi items.parquet đã bị sinh lại.
        """
        if self.sim is None:
            raise RuntimeError("phải gọi fit trước export_similarity")
        app_ids = _app_ids_in_order(items, self.n_items)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        save_npz(p.with_suffix(".npz"), self.sim)
        meta = {
            "model": self.name,
            "k_neighbors": self.k_neighbors,
            "n_items": self.n_items,
            "app_ids": app_ids.tolist(),
        }
        p.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")

    def top_similar_pairs(self, items: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """In vài cặp game giống nhau nhất — để kiểm tra bằng mắt.

        Đây là tiêu chí Xong khi của việc này: các cặp phải trông hợp lý với
        người đọc. Nếu ra toàn cặp vô nghĩa thì có lỗi ở đâu đó.

        items: items.parquet (cần cột i_idx, title). Mỗi cặp chỉ xuất hiện một lần.
        """
        if self.sim is None:
            raise RuntimeError("phải gọi fit trước top_similar_pairs")
        coo = self.sim.tocoo()
        # Bỏ đường chéo (game giống chính nó) và gộp (i, j) với (j, i):
        # ma trận đã cắt K láng giềng nên không đối xứng, giữ giá trị lớn hơn.
        keep = coo.row != coo.col
        i = np.minimum(coo.row[keep], coo.col[keep])
        j = np.maximum(coo.row[keep], coo.col[keep])
        pairs = (
            pd.DataFrame({"i": i, "j": j, "sim": coo.data[keep]})
            .groupby(["i", "j"], as_index=False)["sim"].max()
            .nlargest(n, "sim")
        )
        title = items.set_index("i_idx")["title"]
        pairs["title_i"] = title.reindex(pairs["i"]).to_numpy()
        pairs["title_j"] = title.reindex(pairs["j"]).to_numpy()
        return pairs.reset_index(drop=True)


def load_similarity(path: str, items: pd.DataFrame) -> csr_matrix:
    """Nạp ma trận do ItemKNN.export_similarity xuất ra, KIỂM TRA khớp items.

    Báo lỗi nếu items.parquet hiện tại khác thứ tự app_id lúc huấn luyện —
    nghĩa là A đã chạy lại preprocess và B phải xuất lại (xem PLANB.md).
    """
    p = Path(path)
    meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    _check_saved_app_ids(meta["app_ids"], items, meta["n_items"], p.name, "ma trận tương đồng")
    return load_npz(p.with_suffix(".npz")).tocsr()
