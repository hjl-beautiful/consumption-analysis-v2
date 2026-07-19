"""
RFM 分析模块
目标：基于RFM模型对用户进行分群，识别高价值用户
"""
import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print(" 电商用户消费行为分析 - RFM 分析模块")
print("=" * 60)

# ==================== 1. 加载清洗后的数据 ====================
print("\n【步骤1】加载清洗数据...")
df = pd.read_csv('data/OnlineRetail_cleaned.csv', parse_dates=['InvoiceDate'])
print(f" 数据加载成功: {len(df)} 行")

# ==================== 2. 计算RFM指标 ====================
print("\n" + "=" * 60)
print("【步骤2】计算RFM指标")
print("=" * 60)

# 设定分析时间点（数据集最后一天的后一天）
analysis_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
print(f" 分析基准日期: {analysis_date.date()}")

# 按用户聚合计算RFM
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (analysis_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency
    'Amount': 'sum'  # Monetary
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

print(f"\n RFM指标统计:")
print(rfm.describe())

# ==================== 3. RFM评分（1-5分）====================
print("\n" + "=" * 60)
print("【步骤3】RFM评分")
print("=" * 60)

# 使用分位数进行评分（5分制）
# R越小越好（最近消费），所以分位数要反转
rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])

# 转换为数值
rfm['R_Score'] = rfm['R_Score'].astype(int)
rfm['F_Score'] = rfm['F_Score'].astype(int)
rfm['M_Score'] = rfm['M_Score'].astype(int)

# RFM综合得分
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

print(f"\n RFM评分示例:")
print(rfm.head(10))

# ==================== 4. 用户分群 ====================
print("\n" + "=" * 60)
print("【步骤4】用户分群")
print("=" * 60)

# 定义用户标签
def segment_customer(row):
    if row['RFM_Score'] in ['555', '554', '544', '545', '454', '455', '445']:
        return '重要价值客户'
    elif row['RFM_Score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
        return '重要保持客户'
    elif row['RFM_Score'] in ['512', '511', '422', '421', '412', '411', '311']:
        return '重要挽留客户'
    elif row['RFM_Score'] in ['533', '532', '531', '523', '522', '521', '515', '514', '513', '425', '424', '413', '414', '415', '315', '314', '313']:
        return '重要发展客户'
    elif row['RFM_Score'] in ['155', '154', '144', '214', '215', '115', '114']:
        return '一般价值客户'
    elif row['RFM_Score'] in ['255', '254', '245', '244', '253', '252', '243', '242', '235', '234', '225', '224', '153', '152', '145', '143', '142', '135', '134', '125', '124']:
        return '一般保持客户'
    elif row['RFM_Score'] in ['331', '321', '231', '241', '251']:
        return '一般挽留客户'
    elif row['RFM_Score'] in ['155', '154', '144', '214', '215', '115', '114']:
        return '一般发展客户'
    else:
        return '低价值客户'

rfm['Segment'] = rfm.apply(segment_customer, axis=1)

# 统计各群体
segment_counts = rfm['Segment'].value_counts()
print(f"\n 用户分群统计:")
print(segment_counts)

# ==================== 5. K-Means聚类（基于RFM）====================
print("\n" + "=" * 60)
print("【步骤5】K-Means聚类")
print("=" * 60)

# 标准化RFM值
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# K-Means聚类（K=4）
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# 聚类评估指标（无监督学习用轮廓系数和CH指数，不用准确率）
silhouette = silhouette_score(rfm_scaled, rfm['Cluster'])
ch_score = calinski_harabasz_score(rfm_scaled, rfm['Cluster'])
print(f" 轮廓系数 (Silhouette Score): {silhouette:.4f}  [范围 -1~1，越接近1越好]")
print(f" CH 指数 (Calinski-Harabasz): {ch_score:.2f}  [越大越好]")

# 肘部法则验证 K 值选择
inertias = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit_predict(rfm_scaled)
    inertias.append(km.inertia_)
print(f" 肘部法则 SSE 序列(K=2~8): {[round(x, 2) for x in inertias]}")
print(f" 选定 K=4 (SSE={kmeans.inertia_:.2f})")

# 聚类结果分析
cluster_analysis = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'CustomerID': 'count'
}).round(2)
cluster_analysis.columns = ['平均Recency', '平均Frequency', '平均Monetary', '用户数']

print(f"\n K-Means聚类结果 (K=4):")
print(cluster_analysis)

# 为聚类添加标签
cluster_labels = {
    0: '高价值活跃客户',
    1: '中价值客户',
    2: '低价值沉睡客户',
    3: '新客户'
}
# 根据实际聚类结果动态调整标签


# ==================== 6. 可视化 ====================
print("\n" + "=" * 60)
print("【步骤6】可视化")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 6.1 RFM分布箱线图
rfm_plot = rfm[['Recency', 'Frequency', 'Monetary']].copy()
# 对Monetary取log便于展示
rfm_plot['Monetary_log'] = np.log1p(rfm_plot['Monetary'])

axes[0, 0].boxplot([rfm_plot['Recency'], rfm_plot['Frequency'], rfm_plot['Monetary_log']],
                   labels=['Recency', 'Frequency', 'Monetary(log)'])
axes[0, 0].set_title('RFM指标分布')

# 6.2 用户分群饼图
axes[0, 1].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', startangle=90)
axes[0, 1].set_title('用户分群占比')

# 6.3 K-Means聚类散点图（R vs F）
colors = ['red', 'blue', 'green', 'orange']
for i in range(4):
    cluster_data = rfm[rfm['Cluster'] == i]
    axes[1, 0].scatter(cluster_data['Recency'], cluster_data['Frequency'],
                       c=colors[i], label=f'Cluster {i}', alpha=0.6)
axes[1, 0].set_xlabel('Recency (最近消费天数)')
axes[1, 0].set_ylabel('Frequency (消费频次)')
axes[1, 0].set_title('K-Means聚类 (Recency vs Frequency)')
axes[1, 0].legend()

# 6.4 各群体消费金额对比
segment_monetary = rfm.groupby('Segment')['Monetary'].mean().sort_values(ascending=True)
axes[1, 1].barh(segment_monetary.index, segment_monetary.values, color='skyblue')
axes[1, 1].set_xlabel('平均消费金额 (£)')
axes[1, 1].set_title('各群体平均消费金额')

plt.tight_layout()
plt.savefig('data/rfm_analysis.png', dpi=150, bbox_inches='tight')
print(f" RFM分析图表已保存: data/rfm_analysis.png")
plt.show()

# ==================== 7. 保存结果 ====================
print("\n" + "=" * 60)
print("【步骤7】保存RFM结果")
print("=" * 60)

rfm.to_csv('data/rfm_results.csv', index=False)
print(f" RFM结果已保存: data/rfm_results.csv")

# 保存用户分群统计
segment_summary = rfm.groupby('Segment').agg({
    'CustomerID': 'count',
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'sum']
}).round(2)
segment_summary.to_csv('data/segment_summary.csv')
print(f" 分群统计已保存: data/segment_summary.csv")

# ==================== 8. 输出关键指标 ====================
print("\n" + "=" * 60)
print("【RFM分析完成 - 关键指标汇总】")
print("=" * 60)
print(f" 总用户数: {len(rfm):,}")
print(f" 总消费金额: £{rfm['Monetary'].sum():,.2f}")
print(f" 平均客户价值: £{rfm['Monetary'].mean():.2f}")
print(f" 平均最近消费天数: {rfm['Recency'].mean():.1f} 天")
print(f" 平均消费频次: {rfm['Frequency'].mean():.2f} 次")
print(f"\n 高价值客户占比: {(rfm['Segment'] == '重要价值客户').mean()*100:.1f}%")
print(f" 沉睡客户占比: {(rfm['Segment'] == '低价值客户').mean()*100:.1f}%")
print("=" * 60)
print(" RFM 分析完成！RFM分析结果已保存，准备进入随机森林分类")
print("=" * 60)