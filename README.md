# 金融欺诈检测系统

## 项目概述
本项目是一个基于机器学习的金融欺诈检测系统，集成了XGBoost、MLP神经网络和规则引擎，通过FastAPI提供实时欺诈检测API服务。系统支持Kafka流处理、Prometheus监控和Locust性能测试，具有高扩展性和可维护性。

## 项目结构
```
financial-fraud-detection/
├── data/                       # 数据目录
│   └── historical_transactions.parquet  # 历史交易数据
├── models/                     # 训练好的模型
│   ├── fraud_detector.pkl      # XGBoost模型
│   ├── mlp_model.keras         # MLP模型(v1)
│   ├── mlp_model_v2.keras      # MLP模型(v2)
│   ├── mlp_scaler_v2.pkl       # MLP模型(v2)的标准化器
│   ├── xgboost_pipeline.pkl    # XGBoost模型管道
│   └── xgboost_threshold.pkl   # XGBoost模型阈值
├── notebooks/                  # Jupyter笔记本，因为python之前跑结果太慢了所以用jupyter分步跑，可以加快出结果的速度
│   ├── Untitled.ipynb          
│   └── new.ipynb               
├── reports/                    # 测试报告
│   ├── locust_report_exceptions.csv  # Locust异常报告
│   ├── locust_report_failures.csv    # Locust失败报告
│   ├── locust_report_stats.csv       # Locust统计报告
│   └── locust_report_stats_history.csv  # Locust历史统计
├── src/                        # 源代码
│   ├── api/                    # API服务
│   │   ├── fraud_fastapi.py    # FastAPI主服务
│   │   └── main.py             # 应用入口
│   ├── engine/                 # 规则引擎
│   │   └── rules.py            # 欺诈检测规则
│   ├── ml/                     # 机器学习模型
│   │   ├── dl_mini.py          # MLP模型训练(v1)
│   │   ├── dl_mini_v2.py       # MLP模型训练(v2)
│   │   └── xgb_model.py        # XGBoost模型训练
│   ├── stream/                 # 流处理模块
│   │   ├── kafka_producer.py   # Kafka生产者
│   │   ├── replay_to_kafka.py  # 历史数据重放
│   │   └── streaming.py        # 流处理管道
│   └── utils/                  # 工具函数
│       ├── draw_calculate.py   # 评估指标计算与绘图
│       ├── draw_metrics.py     # 评估指标可视化
│       └── time.py             # 时间处理工具
├── tests/                      # 测试用例
│   └── locustfile.py           # Locust性能测试脚本
├── .bat                        # 批处理脚本
├── .gitattributes              # Git属性配置
├── .gitignore                  # Git忽略配置
├── requirement.txt             # 依赖库列表
└── README.md                   # 项目说明
```

## 核心功能
1. **多模型融合检测**：结合XGBoost和MLP神经网络，提高欺诈检测准确率
2. **规则引擎**：基于durable_rules实现可配置的业务规则检测
3. **实时API服务**：提供低延迟的欺诈检测API接口
4. **流处理支持**：集成Kafka和Apache Beam，支持实时数据流处理
5. **性能测试**：使用Locust进行API性能测试
6. **监控与评估**：提供Prometheus指标和评估报告
7. **数据可视化**：通过Jupyter笔记本进行数据分析和可视化

## 技术栈
- **机器学习**：XGBoost, Keras, Scikit-learn
- **API服务**：FastAPI
- **流处理**：Apache Beam, Kafka
- **规则引擎**：durable_rules
- **性能测试**：Locust
- **监控**：Prometheus, Grafana
- **部署**：Docker, GitHub Actions
- **数据分析**：Jupyter, Pandas, Matplotlib

## 安装与使用

### 环境准备
```bash
pip install -r requirement.txt
```

### 模型训练
```bash
python src/ml/xgb_model.py      # 训练XGBoost模型
python src/ml/dl_mini.py        # 训练MLP模型(v1)
python src/ml/dl_mini_v2.py     # 训练MLP模型(v2)
```

### 启动API服务
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
or （你的环境位置）"D:\AppGallery\Downloads\anaconda\envs\fraud39\python.exe" -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### API调用示例
```python
import requests

data = {
    "V1": -1.359807, "V2": -0.072781, "V3": 2.536347,
    # ... 其他特征
    "Amount": 149.62
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## 性能测试
使用Locust进行API性能测试：
```bash
python -m locust -f tests/locustfile.py --host=http://localhost:8000
```
访问`http://localhost:8089`打开Locust Web界面，设置并发用户数和每秒请求数进行测试。

## 流处理示例
### 启动Kafka生产者
```bash
#在对应位置下先启动 Zookeeper
cd C:\Users\陈一心\kafka_2.13-3.3.1
bin\windows\zookeeper-server-start.bat config\zookeeper.properties
#在对应位置下再启动 Kafka Broker
cd C:\Users\陈一心\kafka_2.13-3.3.1
bin\windows\kafka-server-start.bat config\server.properties
#在本地环境启动kafka
python src/stream/kafka_producer.py
```

### 启动流处理管道
```bash
#在对应位置下启动redis
cd redis
redis-server.exe
#在本地环境启动streaming
python src/stream/streaming.py
```

## 规则引擎
规则引擎配置位于`src/engine/rules.py`使用决策树，支持自定义规则：
```python
@when_all(m.V1 < -5, m.Amount > 1000)
def high_risk_transaction(c):
    c.assert_fact({"risk": "high", "rule": "v1_negative_amount_high"})
```

## 评估指标
- AUC: 0.98+
- 召回率: 78%+
- F1分数: 0.86+

## 监控与可视化
1. 访问`http://localhost:8000/metrics`查看Prometheus指标
2. 使用Jupyter笔记本运行数据分析：`jupyter notebook notebooks/`

## 神经网络尝试
尝试融合一到两层神经网络到模型中，但是并没有显著提升，反而评估结果比原来更差，所以选择不采用。

## 贡献指南
1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/new-feature`)
3. 提交更改 (`git commit -am 'Add new feature'`)
4. 推送分支 (`git push origin feature/new-feature`)
5. 创建Pull Request

## 许可证
本项目采用MIT许可证 - 详见LICENSE文件

## 作者信息
- **Cianna1** - *初始开发者* - [GitHub](https://github.com/Cianna1)
