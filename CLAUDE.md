# CLAUDE.md

復現 Mi, Ma & Toda, arXiv:2606.06200（zero-shot cross-lingual SER）。規則見 README.md 與
`configs/protocol.yaml`（凍結）。目標語言絕不能進 training / sampler / normalization /
early stopping / 超參數選擇；超參數只能看 URDU dev。

## 資料集

### MELD（EN）

音訊前處理與 DataLoader 規則：

- 建 Dataset 前，依 split 與 CSV 標註對應 WAV 路徑；排除缺檔或無法解碼的音檔，存成可用樣本清單
  （`data/manifests/EN.csv`），並記錄被排除的 split、樣本 ID 與原因。
- DataLoader 只讀篩選後的清單。不要在 `__getitem__` 遇到壞檔時回傳 None（會改變 batch 組成）。
- 保留官方 train / val(dev) / test 分組與原始標註。

已知缺檔（`configs/protocol.yaml` 的 `known_missing`）：

| split | 樣本 | 原因 | 狀態 | 來源 |
|---|---|---|---|---|
| train | dia125_utt3 | 原始發行版壞檔，FFmpeg 回報 `moov atom not found`，無法抽出音軌 | 已在本機確認 | https://github.com/declare-lab/MELD/issues/39 |
| val (dev) | dia110_utt7 | 原始 dev 發行版缺檔，與 val 少 1 筆相符 | 待在伺服器上逐筆確認 | https://github.com/declare-lab/MELD/issues/22 |

樣本數（4 類）：我們 train 8,244 / val 896 / test 2,211；論文 Table 1 為 8,245 / 897 / 2,211。

- `xlser.prepare` 只接受「剛好缺上表兩筆」，會自動把預期數量扣掉；缺其他檔照樣報錯並列出 ID。
  因此**不要再用 `--allow-count-mismatch`**，它會放過任何差異。
- 含 EN 的 task（*→DE、*→FR）Table 1 #Samples 預期為 20,164（論文 20,165）。
- 報告結果時註明實際樣本數，以及這兩筆已排除。

轉檔：`bash scripts/meld_mp4_to_wav.sh <MELD.Raw> <meld_wav> [jobs]`（16 kHz mono，失敗清單在 `<meld_wav>/failed.txt`）。

### EMO-DB（DE）

4 類共 339 句，固定 seed 分層切分為 266 / 35 / 38，與論文一致。Upper Bound（3 seeds）：
EN→DE 98.48、CN→DE 94.99、FR→DE 92.82 UAR；論文 97.22 / 97.22 / 95.44。test 只有 38 句，
同設定不同 seed 可差到約 10 分，正式實驗跑 5 個 seed。

### URDU（UR）

300 / 20 / 80，與論文一致。UR 從來不是 target，是唯一允許調超參數的 dev 語料
（lr sweep 選出 1e-3，已凍結於 `configs/train.yaml`）。

### CaFE（FR）

來源是 CAMEO（`~/dataset/CAMEO/CAMEO/data/cafe-00000-of-00001.parquet`，936 句完整版），不是原始
CaFE 資料夾。CAMEO 把原始檔名換成 hash（`file_id`），speaker 與情緒改從 `speaker_id`
（`cafe_01`…`cafe_12`）和 `emotion` 欄位取得，強度（Faible / Fort）資訊不在。

- 轉檔：`python scripts/cameo_to_wav.py <cafe parquet> <out>` → 16 kHz mono wav +
  `<out>/cameo_metadata.csv`；`xlser/corpora/cafe.py` 偵測到這份 CSV 就改讀它。
- 4 類（anger / happiness / sadness / neutral）預期 504 句，分層切成 420 / 42 / 42。
- CAMEO 其他語料（CREMA-D、RAVDESS…）不屬於論文的五個資料集，不使用。

### 尚未準備

ESD（CN，只用 Mandarin 0001–0010）。

### protocol hash

`configs/protocol.yaml` 任何修改都會改變 `protocol_sha256`，`collect_results.py` 會略過舊 hash 的 run。
2026-11 為 MELD 加入 `known_missing` 後，先前的 EMO-DB Upper Bound 與 URDU dev runs 已屬舊 hash
（數字仍記錄在上方與 `configs/train.yaml` 註解）；正式 Table 2 由 `run_all.sh` 全部重跑。
