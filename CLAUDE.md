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
| val (dev) | dia110_utt7 | 原始 dev 發行版缺檔，與 val 少 1 筆相符 | 已在伺服器確認（prepare 列出的缺檔即此兩筆） | https://github.com/declare-lab/MELD/issues/22 |

樣本數（4 類）：我們 train 8,244 / val 896 / test 2,211；論文 Table 1 為 8,245 / 897 / 2,211。

- `xlser.prepare` 只接受「剛好缺上表兩筆」，會自動把預期數量扣掉；缺其他檔照樣報錯並列出 ID。
  因此**不要再用 `--allow-count-mismatch`**，它會放過任何差異。
- 含 EN 的 task（*→DE、*→FR）Table 1 #Samples 預期為 20,164（論文 20,165）。
- 報告結果時註明實際樣本數，以及這兩筆已排除。

轉檔：`bash scripts/meld_mp4_to_wav.sh <MELD.Raw> <meld_wav> [jobs]`（16 kHz mono，失敗清單在 `<meld_wav>/failed.txt`）。
伺服器路徑：`~/dataset/MELD/meld_wav`。manifest `data/manifests/EN.csv` 11,351 筆，split sha256 `8105357fb4f2…`。

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
- 已在伺服器確認：936 句、7 種情緒（anger / disgust / fear / happiness / sadness / surprise 各 144，
  neutral 72）、12 位 speaker 各 78 句；CAMEO 版已是 16 kHz（原始 CaFE 為 48 kHz）。
  輸出在 `~/dataset/CaFE_cameo`，manifest `data/manifests/FR.csv` 504 筆、420 / 42 / 42，
  split sha256 `07655c53309b…`。
- CAMEO 其他語料（CREMA-D、RAVDESS…）不屬於論文的五個資料集，不使用。

### 尚未準備

ESD（CN，只用 Mandarin 0001–0010）。

### protocol hash

`protocol_sha256` 由 `configs/protocol.yaml` 的內容計算（不含註解與 `dev_tasks`），`collect_results.py`
會略過 hash 不同的 run。2026-11-24 改為此算法前的所有 run（含 35 個 *→FR Upper Bound / Baseline 1）
都屬舊 hash；正式 Table 2 由 `run_all.sh` 全部重跑。

## 已知問題：只用 MELD 訓練會崩成全猜 neutral（2026-11-24）

Baseline 1（EN→DE、EN→FR）10 個 run 中 6 個的 **MELD val UAR = 25.0、best_step = 100**：在來源端就只會
猜 neutral（MELD 4 類 neutral 佔 57%，baseline 用一般隨機 batch）。DE / FR 來源的 baseline 正常
（source val 87–100）。判斷依據只用來源端 MELD val，不看 DE / FR 結果。

- 對策候選：`class_weighted_ce`（`configs/train.yaml`，訓練列的反類別頻率，只作用於隨機 batch 的系統）。
- 驗證：`bash scripts/check_meld_training.sh 0 1 2 3`（dev task `EN-EN`：只在 MELD 上訓練 / 選模型 / 評估）。
- 決策規則：依 EN-EN 的 MELD **val** UAR 決定是否開啟，開或不開都套用到所有系統，之後凍結並全部重跑。
- 結果（2026-11-24，5 seeds）：MELD val UAR cw_false 38.6、cw_true 35.5 → **不採用**，`class_weighted_ce: false`。
- 下一個假設：lr 1e-3（在小而平衡的 URDU 上選出）對 MELD 太大。`bash scripts/tune_lr_on_meld.sh 0 1 2 3`
  在 EN-EN 補跑 lr 1e-4 / 3e-4 / 3e-3（1e-3 即 cw_false）。
- 診斷（EN-EN cw_false 訓練曲線）：所有 seed 前 300–600+ 步 ce ≈ 1.10–1.18（= MELD 類別先驗的熵 ≈ 1.12），
  val 停在 25，之後多數爬到 40–43；s3 平台期超過 patience 8 × 100 = 800 步，被提前停止（best_step 100）。
  Baseline 1 的 6 個崩潰 run 同一機制。→ **取消提前停止**：`patience: null`，固定跑 `max_steps`，取 val 最佳
  checkpoint（仍由 val 做 model selection）。依據只來自 MELD 來源端曲線。
- 因訓練規則改變，URDU 與 MELD 的 lr sweep 都在 `patience: null` 下重跑（舊 `runs_dev` 移到 `runs_dev_old`）。
- **lr 重新選擇規則（看結果前固定）**：score(lr) = (EN/CN/DE/FR-UR 的 val UAR 平均 + EN-EN 的 val UAR) / 2，
  兩個 dev 語料各佔一半，取最高者寫入 `configs/train.yaml` 並凍結；不看任何 DE / FR / CN 結果。
  `python scripts/collect_results.py runs_dev --dev --lr-score` 直接算出分數。
