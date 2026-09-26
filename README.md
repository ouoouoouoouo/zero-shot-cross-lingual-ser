# Zero-shot Cross-lingual SER（復現）

復現 Mi, Ma & Toda, *Learning Emotion-discriminative Representations for Zero-Shot
Cross-lingual Speech Emotion Recognition*（arXiv:2606.06200）：wav2vec 2.0 + 語言感知
supervised contrastive loss + 對抗式 speaker classifier（GRL）。

## Zero-shot protocol（固定不動）

所有規則寫在 [`configs/protocol.yaml`](configs/protocol.yaml)。每個 run 會把這個檔案的
sha256 寫進 `run.json`，`scripts/collect_results.py` 會拒收 protocol 不同的 run。

| 階段 | 可以看到的資料（以 EN→DE 為例） |
|---|---|
| 訓練（CE / SupCon / SpkAdv） | EN, CN, FR, UR 的 **train** split |
| Batch sampler | 同上（只拿到 `plan.train`） |
| Speaker classifier 的類別 | 只有訓練語言的 speaker |
| Normalization | 逐句 zero-mean / unit-variance，不算任何語料層級的統計量 |
| Early stopping / model selection | EN, CN, FR, UR 的 **val** split（各語言 UAR 平均） |
| 最終報告 | **只有** German EMO-DB 的 test split（38 句），訓練結束、還原最佳 checkpoint 後才讀取 |

- 目標語言在 zero-shot 系統中不會出現在 train、sampler、speaker map、normalization 或
  early stopping。`xlser/protocol.py` 的 `check_isolation` 會用資料列本身再驗一次
  （語言、語料名稱、檔案路徑、speaker），有違規就丟 `LeakageError`。
- `tests/test_end_to_end.py` 會攔截每一次讀音檔的呼叫，確認 `fit()` 期間沒有開過任何
  EMO-DB 檔案。
- 五個資料集統一成 **neutral / happy / angry / sad**，其他情緒在 `xlser/corpora/*.py`
  掃描時就丟掉。
- 另外附上 `target_all_supplementary`（整個目標語料，EMO-DB 為 339 句）作為補充數字；
  主數字永遠是 target test split，才能跟論文和 Upper Bound 對照。

### 系統

| system | 訓練語言 | sampler | SupCon | SpkAdv |
|---|---|---|---|---|
| `baseline1` | source | random | | |
| `baseline2` | source + non-target | random | | |
| `proposed` | source + non-target | hierarchical | ✓ | ✓ |
| `proposed_no_spk` | source + non-target | hierarchical | ✓ | |
| `proposed_no_supcon` | source + non-target | hierarchical | | ✓ |
| `upper_bound`（非 zero-shot） | target | random | | |

Backbone 永遠是**來源語言**的 wav2vec 2.0（EN: `facebook/wav2vec2-base-960h`,
CN: `TencentGameMate/chinese-wav2vec2-base`, DE: `facebook/wav2vec2-base-de-voxpopuli-v2`,
FR: `facebook/wav2vec2-base-fr-voxpopuli`）。Upper Bound 也用該 task 的來源 backbone
（論文中 EN→DE 與 FR→DE 的 Upper Bound 數字不同，表示如此）。

### 資料切分

| 語言 | 語料 | train / val / test（4 類） | 方式 |
|---|---|---|---|
| EN | MELD | 8245 / 897 / 2211 | 官方切分 |
| CN | ESD（Mandarin 0001–0010） | 11200 / 1400 / 1400 | 依 (speaker, 情緒) 分層 |
| DE | EMO-DB | 266 / 35 / 38 | 依情緒分層 |
| FR | CaFE | 420 / 42 / 42 | 依情緒分層 |
| UR | URDU | 300 / 20 / 80 | 依情緒分層 |

這些 train 數量加總正好等於論文 Table 1 的 #Samples（例如 EN→DE：8245+11200+300+420 = 20165），
`build_plan` 會自動比對 #Samples / #Spk。非官方切分是 `split_seed` 固定的分層隨機切分，
數量與論文完全一致；`prepare` 在數量不符時直接報錯。

## 超參數怎麼選（不能看 target）

`configs/train.yaml` 的超參數所有系統共用，所以**不能看 DE / FR / CN 的任何結果來調**，
否則目標語言會經由超參數間接洩漏。唯一允許調參的地方是 `protocol.yaml` 的 `dev_tasks`：
在 URDU 上跑 Upper Bound（EN-UR / CN-UR / DE-UR / FR-UR，四個 backbone 各一），
因為 UR 從來不是 9 個 task 的 target。

```bash
python -m xlser.prepare --lang UR --root /data/URDU-Dataset
python scripts/inspect_backbones.py --lang UR       # 看各 backbone 最後一層特徵是否幾乎相同
bash scripts/tune_on_urdu.sh 0 1 2 3 4 5            # lr sweep x 4 backbones x 2 seeds
```

依 **mean val UAR** 選一個 lr 寫回 `configs/train.yaml`，之後凍結。每個 run 會記錄
`hparams_sha256`，`collect_results.py` 發現 Table 2 內的 run 用了不同超參數時會警告。

訓練預算用 **optimizer step** 計（`max_steps`，warm-up + cosine），每 `eval_every` 步看一次
val，`patience` 次沒進步就停。用 epoch 計的話，EMO-DB（7 step/epoch）與 20k 句的任務
（560 step/epoch）預算會差 80 倍。

## 使用方式

```bash
pip install -r requirements.txt

# 1) 建 manifest（每個語言一次）
python -m xlser.prepare --lang DE --root /data/emodb          # 含 wav/ 的資料夾
python -m xlser.prepare --lang FR --root /data/CaFE
python -m xlser.prepare --lang UR --root /data/URDU-Dataset
python -m xlser.prepare --lang CN --root /data/ESD
bash scripts/meld_mp4_to_wav.sh /data/MELD.Raw /data/meld_wav
python -m xlser.prepare --lang EN --root /data/meld_wav

# 2) 訓練 + 評估
python -m xlser.train --task EN-DE --system proposed
bash scripts/run_all.sh 0 1 2 3 4 5         # 9 tasks x 6 systems x 3 seeds，一張 GPU 一個 job
python scripts/collect_results.py runs      # Table 2 格式
```

### 在 GPU cluster 上跑

```bash
git clone -b claude/wonderful-wozniak-i2rpae https://github.com/ouoouoouoouo/zero-shot-cross-lingual-ser.git
cd zero-shot-cross-lingual-ser && pip install -r requirements.txt && pytest -q
```

- 計算節點若不能連外網，先在登入節點下載 backbone，再設 `HF_HUB_OFFLINE=1`：
  `python -c "from transformers import Wav2Vec2Model as M; [M.from_pretrained(n) for n in ['facebook/wav2vec2-base-960h','TencentGameMate/chinese-wav2vec2-base','facebook/wav2vec2-base-de-voxpopuli-v2','facebook/wav2vec2-base-fr-voxpopuli']]"`
- manifest 存的是絕對路徑，請在 cluster 上重新跑 `xlser.prepare`；
  `data/manifests/*.split.sha256` 應與其他機器一致。
- 每個 (task, system, seed) 是獨立的 process，可以直接各開一個 job 平行跑。

### Weights & Biases（選用）

```bash
wandb login
python -m xlser.train --task EN-DE --system proposed --wandb [--wandb-project P --wandb-entity E]
```

每 `eval_every` 步記錄 train loss（有啟用的 ce / supcon / spk）、lr、grad norm、pooled 特徵 norm，以及訓練語言的 val UAR；run 以 task 分 group、
system 為 job type，config 中含 protocol sha256 與各階段的資料審計。目標語言的 test UAR / F1
只在訓練結束後寫進 run summary，訓練曲線中不會出現目標語言。`WANDB_MODE=offline` 可離線記錄。

第一步只有 EMO-DB 時：`bash scripts/run_emodb_pipeline.sh /data/emodb` 會建 DE manifest
並跑三個 *→DE 的 Upper Bound（論文：97.22 / 97.22 / 95.44 UAR）。

`data/manifests/<LANG>.split.sha256` 是切分指紋（與絕對路徑無關），建議 commit 起來。

沒有真實資料時可以用合成語料跑通整條 pipeline：

```bash
python scripts/make_dummy_corpora.py data/dummy
python -m xlser.prepare --lang DE --root data/dummy/emodb
python -m xlser.prepare --lang EN --root data/dummy/meld --allow-count-mismatch   # CN/FR/UR 同理
python -m xlser.train --task EN-DE --system proposed --backbone tiny-random --set max_steps=20 eval_every=10
pytest -q
```

## 論文有寫 vs. 我們自己決定的

論文有寫：λ=2.5、α=1.0、β=0.3、N_lang=3、N_cls=4、N_sam=3、mean pooling、
speaker head = GRL→Linear→ReLU→Dropout→Linear、LoRA + bottleneck adapter + weight gating、
各資料集的切分數量。

論文沒寫、放在 [`configs/train.yaml`](configs/train.yaml) 的選擇：τ=0.07、AdamW lr 1e-3（URDU dev sweep 選出）、
最多 3000 step（warm-up 100 + cosine）、每 100 step 驗證、patience 8、LoRA r=8（q/v）、adapter 維度 64、weight gating 解讀為每個
LoRA / adapter 分支乘上可學習的 sigmoid gate、訓練隨機裁切 6 s、baseline 使用一般隨機 batch
（batch size 同為 36）、early stopping 指標為訓練語言 val UAR 的語言平均、非官方切分只依
情緒分層（不做 speaker-independent，因為論文的數量無法做到）。
