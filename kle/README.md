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

## 依赖

- pandas, requests, beautifulsoup4, lxml
