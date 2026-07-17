"""
随机森林分类模块
目标：基于RFM特征预测用户是否为高价值客户，训练可解释的分类模型
"""
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print(" 电商用户消费行为分析 - 随机森林分类模块")
print("=" * 60)

# ==================== 1. 加载RFM数据 ====================
print("\n【步骤1】加载RFM数据...")
rfm = pd.read_csv('data/rfm_results.csv')
print(f" 数据加载成功: {len(rfm)} 行")

# ==================== 2. 定义目标变量 ====================
print("\n" + "=" * 60)
print("【步骤2】定义目标变量")
print("=" * 60)

rfm['ValueScore'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
value_threshold = rfm['ValueScore'].quantile(0.8)
rfm['IsHighValue'] = (rfm['ValueScore'] >= value_threshold).astype(int)

print(f" 高价值客户阈值: ValueScore >= {value_threshold:.1f}")
print(f" 高价值客户数量: {rfm['IsHighValue'].sum()} ({rfm['IsHighValue'].mean()*100:.1f}%)")
print(f" 普通客户数量: {(1-rfm['IsHighValue']).sum()} ({(1-rfm['IsHighValue']).mean()*100:.1f}%)")

# ==================== 3. 特征工程 ====================
print("\n" + "=" * 60)
print("【步骤3】特征工程")
print("=" * 60)

# 基础特征
features = ['Recency', 'Frequency', 'Monetary']

# 衍生特征
rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']  # 平均订单价值
rfm['R_F_Ratio'] = rfm['Recency'] / (rfm['Frequency'] + 1)  # 消费间隔比
rfm['M_F_Ratio'] = rfm['Monetary'] / (rfm['Frequency'] + 1)  # 客单价

feature_cols = features + ['AvgOrderValue', 'R_F_Ratio', 'M_F_Ratio']

X = rfm[feature_cols]
y = rfm['IsHighValue']

print(f" 特征列表: {feature_cols}")
print(f" 特征矩阵: {X.shape}")

# ==================== 4. 划分训练集和测试集 ====================
print("\n" + "=" * 60)
print("【步骤4】划分数据集")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f" 训练集: {len(X_train)} 条 ({len(X_train)/len(X)*100:.0f}%)")
print(f" 测试集: {len(X_test)} 条 ({len(X_test)/len(X)*100:.0f}%)")
print(f" 训练集高价值客户占比: {y_train.mean()*100:.1f}%")
print(f" 测试集高价值客户占比: {y_test.mean()*100:.1f}%")

# ==================== 5. 训练随机森林模型 ====================
print("\n" + "=" * 60)
print("【步骤5】训练随机森林模型")
print("=" * 60)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练模型
rf = RandomForestClassifier(
    n_estimators=100,      # 100棵树
    max_depth=10,          # 最大深度
    min_samples_split=5,   # 最小分裂样本数
    min_samples_leaf=2,    # 最小叶子样本数
    random_state=42,
    n_jobs=-1             # 使用所有CPU核心
)

print(" 开始训练...")
rf.fit(X_train_scaled, y_train)
print(" 训练完成！")

# ==================== 6. 模型评估 ====================
print("\n" + "=" * 60)
print("【步骤6】模型评估")
print("=" * 60)

# 预测
y_pred = rf.predict(X_test_scaled)
y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]

# 计算指标
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f" 准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f" F1-Score: {f1:.4f}")
print(f"\n 分类报告:")
print(classification_report(y_test, y_pred, target_names=['普通客户', '高价值客户']))

# ==================== 7. 可视化 ====================
print("\n" + "=" * 60)
print("【步骤7】可视化")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 7.1 特征重要性
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=True)

axes[0, 0].barh(importance_df['feature'], importance_df['importance'], color='forestgreen')
axes[0, 0].set_xlabel('重要性')
axes[0, 0].set_title('随机森林特征重要性')

# 7.2 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
            xticklabels=['预测:普通', '预测:高价值'],
            yticklabels=['真实:普通', '真实:高价值'])
axes[0, 1].set_title('混淆矩阵')

# 7.3 ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
axes[1, 0].set_xlim([0.0, 1.0])
axes[1, 0].set_ylim([0.0, 1.05])
axes[1, 0].set_xlabel('假阳性率')
axes[1, 0].set_ylabel('真阳性率')
axes[1, 0].set_title('ROC曲线')
axes[1, 0].legend(loc='lower right')

# 7.4 预测概率分布
axes[1, 1].hist(y_pred_proba[y_test==0], bins=30, alpha=0.5, label='普通客户', color='blue')
axes[1, 1].hist(y_pred_proba[y_test==1], bins=30, alpha=0.5, label='高价值客户', color='red')
axes[1, 1].set_xlabel('预测为高价值的概率')
axes[1, 1].set_ylabel('频数')
axes[1, 1].set_title('预测概率分布')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('data/random_forest_analysis.png', dpi=150, bbox_inches='tight')
print(f" 随机森林分析图表已保存: data/random_forest_analysis.png")
plt.show()

# ==================== 8. 保存模型 ====================
print("\n" + "=" * 60)
print("【步骤8】保存模型")
print("=" * 60)

import joblib
joblib.dump(rf, 'data/random_forest_model.pkl')
joblib.dump(scaler, 'data/scaler.pkl')
print(f" 模型已保存: data/random_forest_model.pkl")
print(f" 标准化器已保存: data/scaler.pkl")

# ==================== 9. 输出关键指标 ====================
print("\n" + "=" * 60)
print("【随机森林分类完成 - 关键指标汇总】")
print("=" * 60)
print(f" 准确率: {accuracy*100:.2f}%")
print(f" F1-Score: {f1:.4f}")
print(f" AUC: {roc_auc:.4f}")
print(f"\n 最重要特征: {importance_df.iloc[-1]['feature']} (重要性: {importance_df.iloc[-1]['importance']:.3f})")
print(f" 特征重要性排名:")
for idx, row in importance_df.iterrows():
    print(f"   {row['feature']}: {row['importance']:.3f}")
print("=" * 60)
print(" 随机森林分类完成！随机森林模型已保存，准备进入 Streamlit 部署")
print("=" * 60)