# KLE 快乐8

数据拉取与冷门预测，按分层设计组织。

## 设计 (Layered / 职责分离)

```
kle/
├── main.py              # CLI 入口 (子命令)
├── config/              # 配置层：单一配置源
│   └── settings.py
├── storage/             # 数据访问层：只负责 I/O
│   ├── repository.py   # 读写 CSV (期数倒序)
│   └── fetcher.py      # 从 17500 / 917500 / ydniu 拉取
├── domain/              # 领域层：业务规则
│   └── strategy.py     # UnpopularStrategy (冷号 + 高段偏好)
├── app/                 # 应用层：用例
│   ├── get_data.py     # 用例：拉取并保存
│   └── predict.py      # 用例：预测今日冷门 20 码
├── scripts/
│   └── get_data.py     # 便捷入口，等价于 main.py get-data
└── data/
    └── data.csv        # 开奖数据 (期数大的在前)
```

- **依赖方向**: CLI → app → domain, storage; config 被 app 使用。
- **单一职责**: repository 只做读写，fetcher 只做拉取，strategy 只做打分与选号。

## 使用

```bash
# 拉取数据到 data/data.csv（期数大的在前）
python main.py get-data
# 或
python scripts/get_data.py

# 冷门预测（基于近期冷号 + 高段 61–80 加成）
python main.py predict
python main.py predict --recent 30
```

## Generative Memory Predictor（主研究模型）

核心脚本：`scripts/generative_memory_predictor.py`  
说明：walk-forward 训练 + memory（method weights / attention / pair_success / replay）+ 生成 cover sets。  

### 只预测下一期（不训练）

```bash
# 使用已有 memory 预测下一期（推荐 pick=10）
python scripts/generative_memory_predictor.py --predict-only --load-memory storage/memory_best.npz --n-cover 100 --pick 10

# 不加载 memory（从空 memory 预测，通常更弱）
python scripts/generative_memory_predictor.py --predict-only --n-cover 100 --pick 10
```

### 指定 target 做 walk-forward（训练 + 评估 + 最终预测）

```bash
# 评估 target 附近窗口，并保存 memory
python scripts/generative_memory_predictor.py --target 2026060 --n-warmup 15 --n-eval 20 --n-cover 20 --pick 10 --save-memory storage/memory_2026060.npz

# 使用单一 memory 文件（存在则加载并继续更新，结束后覆盖保存）
python scripts/generative_memory_predictor.py --memory storage/memory.npz --target 2026060 --pick 10
python scripts/generative_memory_predictor.py --memory storage/memory.npz --predict-only --n-cover 100 --pick 10
```

### 预测“下一期 target”（例如 data 最新是 2026063，则 target=2026064）

当 `target == latest + 1` 时，会自动走 prediction-only（因为 CSV 没有实际开奖无法评估）：

```bash
python scripts/generative_memory_predictor.py --target 2026064 --load-memory storage/memory_best.npz --n-cover 100 --pick 10
```

### 快速训练：只用最新 N 期（fast）

```bash
# latest N：target=最新一期、n-eval=N、n-warmup=0
python scripts/generative_memory_predictor.py --latest 100 --n-cover 10 --pick 10 --memory storage/memory.npz
```

### Pool size sweep（同一次运行报告不同 pool 的命中率）

```bash
# 用 max(pool) 生成，然后报告 best-of-k 的命中率（例如 7+ hits）
python scripts/generative_memory_predictor.py --target 2026060 --pick 10 --n-cover 100 --sweep-pool 10,20,50,100 --hit-target 7
```

### Checkpoint（防中断）

```bash
# 每 N step 保存一次（walk-forward / full-dataset 都支持）
python scripts/generative_memory_predictor.py --target 2026060 --pick 10 --n-cover 20 --checkpoint-every 25 --save-memory storage/memory_ckpt.npz
```

### Full dataset（全量 walk-forward）

```bash
python scripts/generative_memory_predictor.py --full-dataset --pick 10 --n-cover 10 --save-memory storage/memory_full.npz
```

## Iterative prediction（分批多 seed 扫描 + 命中统计 + 投入/奖金/ROI）

脚本：`scripts/iterative_predict.py`

```bash
# 多轮迭代（iterations × per-iter）生成 sets，并对比实际开奖，输出 hit 分布与 ROI
python scripts/iterative_predict.py \
  --actual "7,13,16,21,25,31,32,38,45,49,57,58,59,61,63,67,69,72,75,77" \
  --iterations 50 --per-iter 500 \
  --memory storage/memory_best.npz --pick 10

# 如需修改奖金参数：
# --stake 2 --prize5 3 --prize6 5 --prize7 80 --prize8 720 --prize9 8000 --prize10 5000000
```

## Hyperparameter tuning（Optuna，目标 = EV per ticket）

脚本：`scripts/optimize_predictor.py`  
说明：每个 trial 使用 fresh memory（不保存）；最后可以把 best trial 重新跑一遍并保存 memory。

```bash
# 快速搜索（每个 trial 的 warmup/eval 更小，速度更快）
python scripts/optimize_predictor.py --target 2026060 --n-trials 20 --n-cover 20 --epochs 1 --quick

# 保存 best params（JSON） + 保存 best-trial memory（npz）
python scripts/optimize_predictor.py --target 2026060 --n-trials 20 --n-cover 20 --epochs 1 --quick \
  --save-best storage/best_params.json \
  --save-best-memory storage/memory_best.npz

# 一步到位：保存 best memory 后，直接输出下一期预测 N 组
python scripts/optimize_predictor.py --target 2026060 --n-trials 5 --n-cover 20 --epochs 1 --quick \
  --save-best-memory storage/memory_best.npz --predict-after 20

# 若已知实际开奖，可在 --predict-after 后直接算命中与奖金（20 个数用逗号分隔）
python scripts/optimize_predictor.py --target 2026060 --n-trials 5 --n-cover 20 --epochs 1 --quick \
  --save-best-memory storage/memory_best.npz --predict-after 100 \
  --actual "2,4,8,9,14,30,33,35,37,38,40,47,52,59,62,65,73,75,76,79"
```

## Inspect memory file（查看 memory.npz 内容）

脚本：`scripts/inspect_memory.py`

```bash
python scripts/inspect_memory.py storage/memory.npz
python scripts/inspect_memory.py storage/memory_best.npz --brief
```

## 依赖

- pandas, requests, beautifulsoup4, lxml


## 1) Quick walk‑forward training for pick=11
python scripts/generative_memory_predictor.py \
  --target 2026060 \
  --pick 11 \
  --payout none \
  --n-warmup 7 --n-eval 30 --n-cover 50 --epochs 1 \
  --memory-lr 0.2588 --memory-decay 0.9436 \
  --pair-boost-weight 0.1229 --replay-weight 0.1097 \
  --temp-base 0.8674 --replay-min-hits 4 \
  --save-memory storage/memory_pick11.npz
This trains the same model architecture, but with 11 numbers per set. --payout none because we only have kl8_pick10 defined.

## 2) Optimize hyperparameters for pick=11
python -u scripts/optimize_predictor.py \
  --n-trials 30 \
  --pick 11 \
  --payout none \
  --objective ev \
  --quick
This will search best n_warmup/n_eval/memory_lr/... for pick=11.

3) Predict with pick=11
python scripts/generative_memory_predictor.py \
  --predict-only \
  --pick 11 \
  --n-cover 200 \
  --load-memory storage/memory_pick11.npz