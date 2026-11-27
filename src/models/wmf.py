"""WMF / ALS — mô hình chính của RQ1. NGƯỜI B — ngày 8-10.

Đây là mô hình DUY NHẤT trong đồ án dùng được trọng số tin cậy từng ô, nên toàn bộ
thí nghiệm về giờ chơi chạy trên nó.

Hàm mục tiêu (Hu, Koren, Volinsky 2008):

    min_{X,Y}  sum_{u,i} c_ui (p_ui - x_u . y_i)^2 + lambda(||X||^2 + ||Y||^2)

Khác phân rã ma trận cổ điển ở hai chỗ: có trọng số c_ui cho từng ô, và tổng chạy
qua TOÀN BỘ ma trận kể cả ô trống. Ô trống vẫn tham gia huấn luyện với p = 0 nhưng
trọng số chỉ bằng 1, nên mô hình được phép cãi lại giả định "không thích" mà gần
như không bị phạt.
"""

from __future__ import annotations

import json
from pathlib import Path

import implicit
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from .base import BaseRecommender, _app_ids_in_order, _check_saved_app_ids


class WMF(BaseRecommender):
    name = "WMF"

    def __init__(self, n_users: int, n_items: int, factors: int = 64,
                 regularization: float = 0.05, iterations: int = 20,
                 seed: int = 42):
        super().__init__(n_users, n_items)
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.seed = seed
        self.model = None
        # Trung vị confidence lúc huấn luyện: confidence mặc định cho fold_in,
        # để vector người dùng mới cùng thang đo với vector đã học.
        self._conf_median: float | None = None

    def _fit(self, train_df: pd.DataFrame, confidence: np.ndarray | None) -> None:
        """Huấn luyện bằng implicit.als.AlternatingLeastSquares.

        TODO
        ----
        * Dựng ma trận thưa với GIÁ TRỊ LÀ CONFIDENCE, không phải 1.
          C = csr_matrix((confidence, (u_idx, i_idx)), shape=(n_users, n_items))
        * AlternatingLeastSquares(factors=..., regularization=...,
                                  iterations=..., alpha=1.0, random_state=seed)

        CẠM BẪY alpha
        -------------
        Thư viện implicit có tham số `alpha` riêng và một số phiên bản tự nhân ma
        trận đầu vào với nó. Nếu bạn vừa tự tính confidence (đã có alpha bên trong)
        vừa truyền alpha=40 vào constructor thì alpha bị áp dụng HAI LẦN.

        Cách an toàn: tự tính confidence bằng build_confidence của A, rồi đặt
        alpha=1.0. Kiểm tra lại docstring của đúng phiên bản đang cài.

        ĐÃ KIỂM CHỨNG với implicit 0.7.3 (đọc mã nguồn):
        * fit nhân ma trận với alpha khi alpha != 1.0 -> alpha=1.0 thì bỏ qua.
        * Giá trị trong ma trận được dùng ĐÚNG LÀ c_ui: bộ giải cộng c*y_i vào
          vế phải và (c - 1)*y_i y_i^T vào vế trái. Ô trống: p=0, c=1.
        * Ma trận có chiều (n_users, n_items).
        """
        if confidence is None:
            raise ValueError(
                "WMF cần confidence (build_confidence). Không tự đoán trọng số "
                "mặc định để RQ1 không âm thầm chạy sai scheme."
            )
        confidence = np.asarray(confidence, dtype=np.float32)
        if confidence.shape != (len(train_df),):
            raise ValueError(
                f"confidence có shape {confidence.shape}, cần ({len(train_df)},) "
                "— cùng độ dài và thứ tự với train_df"
            )
        if not (confidence >= 1).all():
            raise ValueError("confidence phải >= 1 (implicit dùng c - 1 làm trọng số thêm)")

        C = csr_matrix(
            (confidence, (train_df["u_idx"].to_numpy(), train_df["i_idx"].to_numpy())),
            shape=(self.n_users, self.n_items),
        )
        if C.nnz != len(train_df):
            raise ValueError("train_df có cặp (u_idx, i_idx) trùng: csr_matrix sẽ cộng dồn confidence")

        # implicit tự song song hóa; để OpenBLAS cũng mở nhiều luồng thì hai lớp
        # luồng tranh nhau và chậm hẳn (implicit tự cảnh báo điều này, ngay lúc
        # khởi tạo). Chỉ giới hạn trong khối này, không ảnh hưởng phần còn lại.
        with threadpool_limits(1, "blas"):
            self.model = AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
                alpha=1.0,  # alpha đã nằm trong confidence, xem CẠM BẪY ở trên
                random_state=self.seed,
                use_gpu=False,  # CPU để kết quả lặp lại được giữa các máy
            )
            self.model.fit(C, show_progress=False)
        self._conf_median = float(np.median(confidence))

    def _score_users(self, user_ids: np.ndarray) -> np.ndarray:
        """s(u,i) = x_u . y_i

        Người dùng không có trong train vẫn có hàng trong user_factors nhưng hàng
        đó không học từ dữ liệu nào; BaseRecommender thay bằng độ phổ biến.
        """
        return self.model.user_factors[user_ids] @ self.model.item_factors.T

    def fold_in(self, liked_i_idx, confidence=None) -> np.ndarray:
        """Tính vector cho một người dùng MỚI mà không huấn luyện lại. Dành cho demo.

            x_u = (Y^T C^u Y + lambda I)^-1 Y^T C^u p_u

        confidence: số hoặc mảng cùng độ dài liked_i_idx. None -> trung vị
        confidence lúc huấn luyện (người dùng demo chỉ chọn game, không có giờ chơi).
        Điểm cho mọi game: vector @ item_factors.T (tự loại liked_i_idx).
        """
        if self.model is None:
            raise RuntimeError("phải gọi fit trước fold_in")
        c = self._conf_median if confidence is None else confidence
        return fold_in_vector(self.model.item_factors, self.regularization, liked_i_idx, c)

    def save(self, path: str, items: pd.DataFrame, *, scheme: str, alpha: float,
             cap_hours: float | None = None) -> None:
        """Xuất item_factors và user_factors cho demo. Hạn: ngày 14 (PLANB mục 10).

        Ghi HAI file cạnh nhau (đuôi trong ``path`` bị thay):
          <path>.npz   user_factors, item_factors (np.savez, KHÔNG pickle)
          <path>.json  thiết lập + conf_median + app_ids theo thứ tự i_idx

        scheme / alpha bắt buộc: WMF không tự biết confidence được tính thế nào.
        A nạp bằng load_factors để phát hiện khi items.parquet đã bị sinh lại.
        """
        if self.model is None:
            raise RuntimeError("phải gọi fit trước save")
        app_ids = _app_ids_in_order(items, self.n_items)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            p.with_suffix(".npz"),
            user_factors=np.asarray(self.model.user_factors, dtype=np.float32),
            item_factors=np.asarray(self.model.item_factors, dtype=np.float32),
        )
        meta = {
            "model": self.name,
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "seed": self.seed,
            "scheme": scheme,
            "alpha": float(alpha),
            "cap_hours": cap_hours,
            "conf_median": self._conf_median,
            "n_users": self.n_users,
            "n_items": self.n_items,
            "implicit_version": implicit.__version__,
            "app_ids": app_ids.tolist(),
        }
        p.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")


def fold_in_vector(item_factors: np.ndarray, regularization: float,
                   liked_i_idx, confidence) -> np.ndarray:
    """Vector người dùng mới từ item_factors — KHÔNG cần đối tượng WMF.

    Dựng phương trình giống bộ giải của implicit 0.7.3 (user_linear_equation):
        A = Y^T Y + lambda I + sum_i (c_i - 1) y_i y_i^T
        b = sum_i c_i y_i
    rồi giải x = A^-1 b. Dùng được sau load_factors (lấy regularization và
    conf_median trong meta).
    """
    Y = np.asarray(item_factors, dtype=np.float64)
    liked = np.asarray(liked_i_idx, dtype=np.int64).ravel()
    if liked.size == 0:
        raise ValueError("liked_i_idx rỗng: người dùng phải chọn ít nhất một game")
    if len(np.unique(liked)) != len(liked):
        raise ValueError("liked_i_idx có game trùng")
    if liked.min() < 0 or liked.max() >= len(Y):
        raise ValueError(f"liked_i_idx phải trong [0, {len(Y)})")
    c = np.broadcast_to(np.asarray(confidence, dtype=np.float64), liked.shape)
    if (c < 1).any():
        raise ValueError("confidence phải >= 1")
    Yl = Y[liked]
    A = Y.T @ Y + regularization * np.eye(Y.shape[1]) + (Yl.T * (c - 1)) @ Yl
    b = Yl.T @ c
    return np.linalg.solve(A, b).astype(np.float32)


def load_factors(path: str, items: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """Nạp file do WMF.save xuất ra, KIỂM TRA khớp items. Trả về
    (user_factors, item_factors, meta).

    Báo lỗi nếu items.parquet hiện tại khác thứ tự app_id lúc huấn luyện.
    """
    p = Path(path)
    meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    _check_saved_app_ids(meta["app_ids"], items, meta["n_items"], p.name, "factors của WMF")
    with np.load(p.with_suffix(".npz"), allow_pickle=False) as f:
        return f["user_factors"], f["item_factors"], meta
