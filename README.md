# 电商用户消费行为分析系统

> 基于 RFM 模型与机器学习的用户价值洞察平台 | UCI Online Retail Dataset

## 项目概述

本项目基于 UCI Online Retail 数据集，构建完整的电商用户消费行为分析流程：

- **数据清洗与预处理**：处理缺失值、异常值，构建用户-订单宽表
- **RFM 分析**：基于最近消费时间（Recency）、消费频率（Frequency）、消费金额（Monetary）进行用户价值分层
- **K-Means 聚类**：对用户进行价值分群，识别高价值/潜力/流失用户
- **随机森林分类**：预测用户是否为高价值客户，支持特征选择与交互训练

## 文件结构

```
.
├── app.py                      # Streamlit 可视化主程序
├── data_cleaning.py            # 数据清洗与预处理
├── rfm_analysis.py             # RFM 分析与 K-Means 聚类
├── random_forest.py            # 随机森林分类模型
├── data/                       # 数据集与中间结果
│   ├── OnlineRetail_cleaned.parquet
│   ├── rfm_results.csv
│   └── ...
└── requirements.txt
```

## 技术栈

- Python 3.10+
- Pandas / NumPy
- scikit-learn（K-Means、Random Forest、StandardScaler）
- Matplotlib / Seaborn
- Streamlit

## 运行方式

按顺序执行：

```bash
python data_cleaning.py
python rfm_analysis.py
python random_forest.py
streamlit run app.py
```

## 数据说明

- **来源**: UCI Machine Learning Repository — Online Retail Dataset
- **内容**: 英国某电商 2010-2011 年在线交易记录
- **用途**: 用户行为分析、RFM 分层、客户价值预测

## 项目定位

本项目为**用户增长与价值分析**方向的机器学习项目，重点展示：
- 从原始交易数据到用户特征工程的完整流程
- RFM 经典用户分层方法
- 聚类与分类模型的实际应用
- Streamlit 交互式可视化部署
