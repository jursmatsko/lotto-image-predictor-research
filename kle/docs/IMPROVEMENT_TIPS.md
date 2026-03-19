# 预测效果改进建议

## 1. 数据已修复
- **期号错误**：`data.csv` 第 2 行期号 `0662026066` 已改为 `2026066`

## 2. 避免覆盖表现好的 memory
全量训练会**覆盖** memory，可能让之前表现好的状态变差。

**做法：**
```bash
# 备份当前表现好的 memory
cp storage/memory_full.npz storage/memory_backup_$(date +%Y%m%d).npz

# 全量训练用单独文件，预测用备份
python scripts/generative_memory_predictor.py --full-dataset --memory storage/memory_train.npz
python scripts/generative_memory_predictor.py --predict-only --memory storage/memory_backup_*.npz
```

## 3. 评估时保证期号一致
`--actual` 必须对应**预测目标期**的开奖号。

- 若数据最新期是 2026033，则预测的是 **2026034**
- 此时 `--actual` 应为 2026034 的开奖号

```bash
# 预测 2026034，用 2026034 实际开奖号评估
python scripts/generative_memory_predictor.py --predict-only \
  --pick 10 --payout kl8_pick10 --n-cover 100 \
  --filter-top 100 --overgen-factor 15 --prioritize-8plus \
  --memory storage/memory_full.npz \
  --actual "4,6,19,25,26,29,30,31,32,43,44,48,50,62,65,66,68,71,72,80"
```

## 4. 回测指定期号
用 walk-forward 得到某期的预测，再与该期实际对比：

```bash
python scripts/generative_memory_predictor.py --target 2026032 \
  --pick 10 --payout kl8_pick10 --n-cover 100 \
  --filter-top 100 --overgen-factor 15 --prioritize-8plus \
  --n-warmup 20 --n-eval 10 --epochs 5 \
  --memory storage/memory_full.npz \
  --actual "2,4,9,22,24,26,27,28,31,34,36,39,43,44,53,55,59,62,65,70"
```

## 5. 全量训练时启用 prioritize-8plus
训练阶段也按 8+ 潜力排序，有利于学到高命中模式：

```bash
python scripts/generative_memory_predictor.py --full-dataset \
  --n-cover 50 --epochs 10 --pick 10 \
  --payout kl8_pick10 --prioritize-8plus \
  --memory storage/memory_full.npz
```

## 6. 推荐预测命令（沿用昨晚表现好的配置）
```bash
python scripts/generative_memory_predictor.py --predict-only \
  --pick 10 --payout kl8_pick10 \
  --n-cover 100 --filter-top 100 --overgen-factor 15 \
  --prioritize-8plus \
  --memory storage/memory_full.npz
```

## 7. 若昨晚结果更好，可能原因
- 当时 memory 尚未被全量训练覆盖
- 数据或期号与当前不同
- 建议：在跑全量训练前先备份 `memory_full.npz`
