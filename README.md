# 电商用户消费行为分析系统

> 基于 RFM 模型与机器学习的用户价值洞察平台 | UCI Online Retail Dataset

## 项目概述

本项目基于 UCI Online Retail 数据集，构建完整的电商用户消费行为分析流程：

- **数据清洗与预处理**：处理缺失值、异常值，构建用户-订单宽表
- **RFM 分析**：基于最近消费时间（Recency）、消费频率（Frequency）、消费金额（Monetary）进行用户价值分层
- **K-Means 聚类**：对用户进行价值分群，识别高价值/潜力/流失用户，使用**轮廓系数（Silhouette Score）和 CH 指数**评估聚类质量（无监督学习不用准确率）
- **随机森林分类**：基于 RFM 特征预测用户是否为高价值客户，使用**准确率、F1-Score、AUC** 评估分类性能（监督学习才用准确率）

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
- 聚类（无监督）与分类（监督）两类模型的区分应用与对应评估指标
- Streamlit 交互式可视化部署

## 模型评估说明

| 模型 | 类型 | 评估指标 | 说明 |
|---|---|---|---|
| K-Means | 无监督聚类 | 轮廓系数、CH 指数、肘部法则 SSE | 无监督学习**没有标签**，不能用准确率 |
| 随机森林 | 监督分类 | 准确率、F1-Score、AUC、混淆矩阵 | 监督学习有真实标签，可用准确率 |

> 常见误区：把 K-Means 的结果用"准确率"描述是错误的。本项目用轮廓系数评估聚类质量，用准确率评估随机森林分类性能。

### 实际运行结果

| 指标 | 数值 | 说明 |
|---|---|---|
| 轮廓系数 (Silhouette) | **0.6162** | >0.5 表示聚类结构合理 |
| CH 指数 | 3149.72 | 类间差异大、类内差异小 |
| 随机森林准确率 | 98.7% | 监督分类性能 |
| 随机森林 AUC | 0.99 | 区分能力强 |

### K-Means 聚类结果（4 群）

| 分群 | 用户数 | 平均 R(天) | 平均 F(次) | 平均 M(£) | 业务含义 |
|---|---|---|---|---|---|
| 高价值用户 | 13 | 7.38 | 82.54 | 127,338 | 高频高消费核心客户 |
| 潜力用户 | 204 | 15.50 | 22.33 | 12,709 | 中频高客单，待激活 |
| 一般用户 | 3,054 | 43.70 | 3.68 | 1,359 | 低频普通消费，量大 |
| 流失用户 | 1,067 | 248.08 | 1.55 | 481 | 长期未复购，需召回 |
