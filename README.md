# Fraud Detection API with XGBoost + MLP + Rules Engine

## 📌 项目简介
本项目为一个基于 **信用卡交易数据** 的金融欺诈检测系统，集成了多模型预测（XGBoost 与 MLP 融合）、规则引擎（durable_rules）与实时 API 服务（FastAPI）。系统具有良好的可扩展性与可视化监控能力，并支持未来迁移至流处理架构（如 GCP Dataflow + Pub/Sub）。

---

## 🎯 项目目标
- 精准识别高风险欺诈交易（Recall > 78%，AUC > 0.98）
- 将模型预测与规则推理融合，提升泛化与可解释性
- 构建 API 服务端口，支持实时接入与限流保护
- 支持 Prometheus 监控与 GitHub Actions 自动测试

---

## 🏗 项目架构

```
                        ┌──────────────┐
                        │  Client/Post │
                        └──────┬───────┘
                               ↓
                         ┌─────────────┐
                         │   FastAPI   │
                         ├─────────────┤
                         │XGBoost Model│ ← fraud_detector.pkl
                         │   +         │
                         │ MLP Model   │ ← mlp_model.keras + scaler.pkl
                         ├─────────────┤
                         │ Rule Engine │ ← durable_rules + tree规则
                         └──────┬──────┘
                                ↓
                        ┌──────────────┐
                        │   Response   │
                        └──────────────┘
```

---

## 🧠 模型融合策略
- **XGBoost**：以 Class 标签训练，设定 scale_pos_weight 平衡类别不均衡；通过验证集确定最佳阈值（F1 最优）
- **MLP**：构建 2–3 层简单全连接神经网络（Keras），并用 `StandardScaler` 标准化输入特征
- **融合方式**：
  ```python
  final_prob = 0.7 * prob_xgb + 0.3 * prob_mlp
  ```

---

## 🧾 规则引擎集成
基于 `durable_rules` 实现，涵盖：
- 金额阈值规则（高风险大额交易）
- 特征组合规则（如 V17 ↑ 且 V10 ↓）
- 样本导出的决策树规则翻译（共 14 条）

规则命中信息通过 `rule_engine_result` 字段返回。

---

## 📈 模型评估指标
- XGBoost AUC: **0.9802**
- MLP AUC: **0.9724 ~ 0.9774**（2层 / 3层）
- 最终融合 F1: **0.8603**

---

## 🚀 接口调用示例
```bash
POST /predict
{
  "V1": -1.23, ..., "Amount": 100.0
}
```
返回结果：
```json
{
  "is_fraud": 1,
  "final_fraud_probability": 0.862513,
  "risk_level": "high",
  "prob_xgb": 0.89,
  "prob_mlp": 0.77,
  "rule_engine_result": {"rule": "high_amount", "risk": "medium"}
}
```

---

## 📦 模块说明
| 模块 | 功能 |
|------|------|
| `model_api.py` | FastAPI 主服务，融合模型预测与规则引擎 |
| `rules.py`     | durable_rules 规则定义（金额、组合、树分支） |
| `dl_mini.py`   | Keras MLP 快速建模与评估脚本 |
| `kafka_producer.py` / `replay_to_kafka.py` | Kafka 交易流模拟器 |
| `streaming.py` | Beam 实时流管道（消费 Kafka → 调用 API → Redis 存储） |
| `draw_calculate.py` | 支持 BigQuery 回测后指标绘图 |

---

## 🛡️ 系统增强功能
- `Prometheus` 指标暴露：API 请求量、延迟等（`/metrics`）
- `slowapi` 限流防护：默认每个 IP 每分钟 ≤ 10 次请求
- `tenacity` 重试机制：对模型调用失败自动尝试 3 次
- `Pub/Sub + Beam + Redis` 支持未来迁移至 GCP 流处理

---

## 🧪 后续可扩展方向
- 使用 Google Pub/Sub 推送实时交易 → Dataflow 调用本服务 → Redis 存储 → BigQuery/Looker 监控准确率
- 增加更多模型（如 LightGBM、Autoencoder）、置信度融合机制
- 增加 Flask-Limiter / circuit breaker / 配置化动态规则系统

---


## 📦 Installation

To install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 📚 Dataset

* **Credit Card Fraud Detection | Kaggle**: [Link to dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* Download the dataset and place it in the appropriate directory.

---

## 🧑‍💻 Author
* **Cianna1**
- 陈一心 | 智能风控实战 / 模型融合与实时欺诈检测
- GitHub 项目地址：[https://github.com/Cianna1/financial-fraud-detection](https://github.com/Cianna1/financial-fraud-detection)



