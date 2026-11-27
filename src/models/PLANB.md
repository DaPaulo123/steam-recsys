# Phần B — Mô hình | Đồ án Hệ gợi ý game Steam (IT4613)

## Ngữ cảnh
- Dữ liệu: Game Recommendations on Steam (Kaggle). Không có rating 1–5 → bài toán xếp hạng top-N.
- Luận điểm chính (RQ1): giờ chơi làm trọng số tin cậy có tốt hơn tín hiệu nhị phân không, hàm nào tốt nhất. B chạy, A cấp tín hiệu.
- RQ2 (tùy chọn, B): chuẩn hóa giờ chơi theo trung vị từng game — chỉ làm nếu RQ1 xong trước ngày 15.
- B sở hữu: `src/models/`, `src/experiments/`. Chương báo cáo: "Bốn mô hình và thiết lập thí nghiệm".

## Quy tắc bắt buộc
- Chỉ sửa file trong `src/models/`, `src/experiments/`. File của người khác → nhắn chủ sở hữu, không tự sửa.
  - A: `src/data/`, `src/content/`, `data/`, `demo/` | C: `src/eval/`, `reports/`
- Không tự đổi hợp đồng giao diện (tên cột, chữ ký hàm). Cần đổi → báo cả nhóm.
- Git: làm trên nhánh `feat/models`, gộp vào `main` ở buổi họp tuần. Commit theo conventional commits, chia theo khối ý nghĩa.
- Không commit `data/`, `results/` chứa dữ liệu lớn, `.env`, file cấu hình cá nhân, file mô hình lớn.
- Cố định seed. Mỗi lần chạy ghi 1 dòng vào `results/runs.csv` kèm siêu tham số + seed.
- Kẹt > 4 tiếng → báo nhóm.

## Hợp đồng giao diện (đã chốt)
Dữ liệu:
- `data/splits/{train,valid,test}.parquet`: `u_idx` int32, `i_idx` int32, `hours` float32, `ts` int64 (chỉ tương tác dương)
- `data/interim/items.parquet`: i_idx, app_id, title, date_release, price_final, win, mac, linux, tags (nối |), description
- `data/interim/maps.json`: n_users, n_items, cutoff_train, cutoff_valid, seed

Hàm:
```python
# A — src/data/signals.py
def build_confidence(df, scheme="log", alpha=40.0, cap_hours=100.0) -> np.ndarray:
    """scheme ∈ {binary, log, linear, capped}; cùng độ dài và thứ tự với df"""

# B — src/models/*.py
class Recommender(Protocol):
    name: str
    def fit(self, train_df, confidence=None) -> None: ...   # mô hình không dùng trọng số thì bỏ qua confidence, không báo lỗi
    def recommend(self, user_ids, k=10, exclude_seen=True) -> np.ndarray: ...
        # shape (len(user_ids), k), chứa i_idx đã xếp hạng, KHÔNG có -1 hay NaN

# C — src/eval/*.py
def evaluate(preds, truth, k_list=(10, 20), item_pop=None, n_items=None, n_users=None) -> dict: ...
def run_eval(model, train_df, test_df, k_list=(10, 20), item_subset=None) -> dict: ...
```
Lưu ý: `contracts.py` và `make_fake.py` đã có trong repo — đọc trước khi code.

## Kiểm tra trên dữ liệu giả (seed 42)
- ItemKNN cosine đúng → nDCG@10 ≈ 0,25. MostPopular ≈ 0,04.
- Mô hình không bỏ xa MostPopular trên dữ liệu giả → lỗi ở code, không phải dữ liệu.

---

## Giai đoạn 1 — Tuần 1: dựng nền
- [x] **1. Nhánh** — cập nhật `main`, tạo `feat/models`.
  Dùng fork: `origin` = fork của B, `upstream` = repo nhóm. Cập nhật `main` = `git pull upstream main`.
- [x] **2. Khung Recommender + cài `implicit` (ngày 1–2)**
  `base.py` lo: nhớ lịch sử cho `exclude_seen`, dự phòng theo độ phổ biến (đếm từ train) cho người dùng không có trong train.
  Xong khi: mô hình rỗng trả mảng **int32** đúng shape `(len(user_ids), k)`, không -1/NaN, **kể cả với người dùng không có trong train**.
  Lưu ý: `data_fake` không có người dùng mới (cả 500 người đều có trong train) → phải giả lập bằng cách bỏ người khỏi train mới thử được dự phòng.
- [x] **3. MostPopular (ngày 3)** — đếm tương tác mỗi game TỪ TẬP TRAIN, không dùng cột user_reviews.
  Xong khi: top-10 hợp lý trên dữ liệu giả.
  Kết quả (data_fake): nDCG@10 = 0,041 (tự tính tạm, chờ `run_eval` của C).
- [ ] **4. ItemKNN (ngày 4–5)** — `implicit` CosineRecommender, quét K ∈ {20, 50, 100}.
  Xong khi: in được vài cặp game tương đồng cao nhất và trông hợp lý; nDCG@10 ≈ 0,25 trên dữ liệu giả (cần `run_eval` của C).
  Quyết định: ItemKNN BỎ QUA `confidence`, tính tương đồng trên ma trận nhị phân (đúng công thức |U_i ∩ U_j| trong `knn.py`).
  Trạng thái: xong trên data_fake — nDCG@10 = 0,248 / 0,251 / 0,250 (K = 20/50/100, tự tính tạm); 15/15 cặp đầu cùng nhóm. Còn: đối chiếu bằng `run_eval` của C; chọn K trên valid thật (mục 6).
- [x] **5. Xuất ma trận tương đồng ItemKNN ra `models/`** cho demo của A (A cần từ ngày 11).
  Lưu kèm danh sách `app_id` theo thứ tự `i_idx` đã dùng khi huấn luyện, để A kiểm tra khớp với `items.parquet` lúc nạp.
  Cách dùng: `ItemKNN.export_similarity(path, items)` ghi `<path>.npz` + `<path>.json`; A nạp bằng `load_similarity(path, items)` (báo lỗi nếu items.parquet đã bị sinh lại). → Báo A.
- [ ] **6. Ngày 6–7 (họp đồng bộ):** chạy MostPopular + ItemKNN trên `splits/` thật, ra nDCG@10 cả nhóm tin được.

## Giai đoạn 2 — Tuần 2: thí nghiệm RQ1
- [x] **7. Stub confidence (ngày 8)** — viết stub `build_confidence` trong thư mục của B, đúng chữ ký hợp đồng, trả mảng cùng độ dài/thứ tự với df. Khi A xong: đổi import sang `src.data.signals`, xóa stub.
  Ở `src/experiments/confidence_stub.py`.
- [ ] **8. WMF/ALS (ngày 8–10)** — `AlternatingLeastSquares(factors=64, regularization=0.05, iterations=20, alpha=1.0)`. KHÔNG tinh chỉnh siêu tham số nào ngoài alpha của build_confidence.
  ĐÃ KIỂM CHỨNG (đọc mã nguồn implicit 0.7.3): `fit` tự nhân ma trận với alpha khi alpha ≠ 1 → đặt alpha=1.0 là đúng; giá trị ma trận được dùng đúng là c_ui (bộ giải dùng c − 1); ma trận chiều (n_users, n_items).
  Seed: `random_state` lấy từ `config/default.yaml`, không gõ cứng trong code (WMF nhận tham số `seed`, script thí nghiệm đọc config rồi truyền vào).
  Xong khi: chạy hết trên dữ liệu thật < 10 phút và vượt MostPopular trên nDCG@10; chạy cùng cấu hình 2 lần ra nDCG giống hệt.
  Trạng thái: xong trên data_fake — vượt MostPopular (0,137 vs 0,041), 2 lần chạy giống hệt từng bit; OpenBLAS giới hạn 1 luồng trong lúc fit. Còn: chạy dữ liệu thật đo thời gian.
  ⚠ Theo dõi: trên data_fake WMF (0,137) kém xa ItemKNN (0,25). Nghi do siêu tham số cố định quá lớn so với bộ giả nhỏ — CHƯA kiểm chứng; không tinh chỉnh (trái quy tắc), xem lại trên dữ liệu thật.
  `fold_in` và `save` chưa làm (thuộc demo / mục 10).
- [ ] **9. Thí nghiệm RQ1 (ngày 11–14)** — 4 scheme {binary, log, linear, capped} × alpha {1, 10, 40, 100} = 16 lần chạy. Chọn trên valid, báo cáo test MỘT lần.
  1. Quét 16 cấu hình: học trên `train`, chấm trên `valid`.
  2. Chọn cấu hình thắng (nếu đỉnh rơi vào biên alpha → mở rộng dải, chạy tiếp).
  3. Gộp `train + valid`, TÍNH LẠI confidence trên tập gộp, học lại với cấu hình thắng.
  4. Chấm `test` một lần, truyền tập gộp làm `train_df` vào `run_eval`.
  ⚠ Chốt với C trước ngày 11: đồng ý cách học lại trên train+valid để chấm test (và RQ3 làm giống vậy).
  Mỗi lần chạy lưu kèm bản sao `config/default.yaml` vào `results/`.
  (Cần nhóm đồng ý) Chạy cấu hình nhất và nhì với 3 seed để đo nhiễu; báo cáo kèm độ dao động.
  Xong khi: `results/runs.csv` đủ 16 dòng kèm siêu tham số + seed.
- [ ] **10. Lưu mô hình tốt nhất (ngày 14)** — xuất `item_factors`, `user_factors` của cấu hình thắng ra `models/`.
  Lưu kèm danh sách `app_id` theo thứ tự `i_idx` đã dùng khi huấn luyện, để A kiểm tra khớp với `items.parquet` lúc nạp.
  Định dạng: hai mảng numpy + `.json` ghi thiết lập (factors, regularization, scheme, alpha, seed). Không pickle mô hình `implicit`.
  ⚠ Chốt với A: demo dùng ItemKNN cho người dùng mới, ALS cho người dùng cũ?
  Xong khi: A nạp được và sinh gợi ý không cần huấn luyện lại.
  Trạng thái: code xong, kiểm tra trên data_fake. `WMF.save(path, items, scheme=, alpha=)` → `.npz` + `.json`; A nạp bằng `load_factors(path, items)`. Người dùng mới (nếu A muốn dùng ALS): `fold_in_vector(item_factors, meta["regularization"], liked, meta["conf_median"])` — khớp 100% top-10 với `recalculate_user` của implicit. Còn: chạy với cấu hình thắng RQ1 (ngày 14).
- [ ] **11. Ngày 14 (họp):** chốt cấu hình thắng RQ1 + quyết định dứt khoát có làm RQ2 không.

## Giai đoạn 3 — Tuần 3–4
- [ ] **12. RQ2 (ngày 15–17, tùy chọn)** — chạy lại cấu hình thắng với giờ chơi chuẩn hóa theo trung vị từng game. Bỏ qua nếu RQ1 chưa xong trước ngày 15. Nếu không làm → hỗ trợ C chạy RQ3.
- [ ] **13. Chương mô hình (ngày 18–25)** — bốn mô hình, công thức, thiết lập thí nghiệm, kết quả RQ1.
- [ ] **14. Ngày 21 (họp):** khóa số liệu — sau đó không chạy thêm thí nghiệm.
- [ ] **15. Slide phần mô hình** — ⚠ mốc chưa thống nhất (mục 7 ghi ngày 19–20, lịch mục 5 ghi 26–27) → hỏi C.
- [ ] **16. Ngày 28:** chạy thử buổi bảo vệ; trước đó giảng lại phần mô hình cho A và C.

## Phụ thuộc cần canh
- Mục 4 cần `run_eval` của C (ngày 4–5).
- Mục 8 cần `build_confidence` của A (ngày 8–10) → dùng stub mục 7.
- A phụ thuộc B: ma trận tương đồng (mục 5), factors (mục 10).
- A chạy lại `preprocess.py` → B phải xuất lại ma trận tương đồng và factors.
- Đề xuất nhóm cố định đúng một phiên bản `implicit` trong `requirements.txt` (hiện là `>=0.7.2`; file không thuộc B).

## Ngoài phạm vi (ghi vào "hướng phát triển", không làm)
BPR, lai ghép, mô hình đồ thị, tín hiệu tiêu cực tường minh.