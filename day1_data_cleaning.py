"""
Day 1: 数据清洗与预处理
目标：将原始交易数据清洗为可用于RFM分析和建模的干净数据
"""

import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示（Windows）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("电商用户消费行为分析 - Day 1: 数据清洗")
print("=" * 60)

# ==================== 1. 加载数据 ====================
print("\n【步骤1】加载原始数据...")
df = pd.read_excel('data/Online Retail.xlsx')

print(f" 数据加载成功！")
print(f"   数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"\n字段信息:")
print(df.dtypes)
print(f"\n前5行预览:")
print(df.head())

# ==================== 2. 数据质量评估 ====================
print("\n" + "=" * 60)
print("【步骤2】数据质量评估")
print("=" * 60)

# 缺失值统计
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    '缺失数量': missing,
    '缺失比例(%)': missing_pct
})
print(f"\n 缺失值统计:")
print(missing_df[missing_df['缺失数量'] > 0])

# 重复值
duplicates = df.duplicated().sum()
print(f"\n 重复行数量: {duplicates}")

# 基本统计
print(f"\n 数值型特征统计:")
print(df.describe())

# ==================== 3. 数据清洗 ====================
print("\n" + "=" * 60)
print("【步骤3】数据清洗")
print("=" * 60)

# 3.1 删除CustomerID缺失的记录（无法识别用户）
print("\n🧹 删除CustomerID缺失的记录...")
before_drop = len(df)
df = df.dropna(subset=['CustomerID'])
after_drop = len(df)
print(f"   删除前: {before_drop} 行")
print(f"   删除后: {after_drop} 行")
print(f"   删除比例: {(before_drop - after_drop) / before_drop * 100:.2f}%")

# 3.2 删除Quantity和UnitPrice为负数的记录（退货和取消订单）
print("\n🧹 处理退货和取消订单...")
print(f"   Quantity < 0 的记录: {(df['Quantity'] < 0).sum()} 条")
print(f"   UnitPrice < 0 的记录: {(df['UnitPrice'] < 0).sum()} 条")

# 保留正常交易（Quantity > 0 且 UnitPrice > 0）
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
print(f"   清洗后剩余: {len(df)} 条正常交易记录")

# 3.3 计算交易金额
print("\n💰 计算交易金额 (Amount = Quantity × UnitPrice)...")
df['Amount'] = df['Quantity'] * df['UnitPrice']
print(f"   交易金额统计:")
print(df['Amount'].describe())

# 3.4 转换日期格式
print("\n 处理日期字段...")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
print(f"   日期范围: {df['InvoiceDate'].min()} 至 {df['InvoiceDate'].max()}")

# 3.5 提取时间特征
print("\n 提取时间特征...")
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df['Hour'] = df['InvoiceDate'].dt.hour
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek  # 0=周一, 6=周日

print(f"   年份分布:\n{df['Year'].value_counts().sort_index()}")

# ==================== 4. 保存清洗后的数据 ====================
print("\n" + "=" * 60)
print("【步骤4】保存清洗后的数据")
print("=" * 60)

# 保存完整清洗数据
df.to_csv('data/OnlineRetail_cleaned.csv', index=False)
print(f" 清洗数据已保存: data/OnlineRetail_cleaned.csv")

# 保存数据清洗报告
with open('data/cleaning_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("数据清洗报告\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"原始数据: {before_drop} 行\n")
    f.write(f"清洗后数据: {len(df)} 行\n")
    f.write(f"数据保留率: {len(df) / before_drop * 100:.2f}%\n\n")
    f.write("清洗操作:\n")
    f.write("1. 删除CustomerID缺失记录\n")
    f.write("2. 删除退货和取消订单(Quantity/UnitPrice < 0)\n")
    f.write("3. 计算交易金额Amount\n")
    f.write("4. 转换日期格式并提取时间特征\n")
    f.write(f"\n日期范围: {df['InvoiceDate'].min()} 至 {df['InvoiceDate'].max()}\n")
    f.write(f"唯一用户数: {df['CustomerID'].nunique()}\n")
    f.write(f"唯一商品数: {df['StockCode'].nunique()}\n")
    f.write(f"唯一订单数: {df['InvoiceNo'].nunique()}\n")

print(f" 清洗报告已保存: data/cleaning_report.txt")

# ==================== 5. 快速可视化 ====================
print("\n" + "=" * 60)
print("【步骤5】快速可视化")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 月度交易趋势
monthly_sales = df.groupby(['Year', 'Month'])['Amount'].sum().reset_index()
monthly_sales['YearMonth'] = monthly_sales['Year'].astype(str) + '-' + monthly_sales['Month'].astype(str).str.zfill(2)

axes[0, 0].plot(range(len(monthly_sales)), monthly_sales['Amount'], marker='o')
axes[0, 0].set_title('月度销售额趋势')
axes[0, 0].set_xlabel('月份')
axes[0, 0].set_ylabel('销售额')
axes[0, 0].tick_params(axis='x', rotation=45)

# 5.2 交易金额分布（取log处理长尾）
axes[0, 1].hist(np.log1p(df['Amount']), bins=50, color='skyblue', edgecolor='black')
axes[0, 1].set_title('交易金额分布(log变换)')
axes[0, 1].set_xlabel('log(Amount + 1)')
axes[0, 1].set_ylabel('频数')

# 5.3 各国家订单量Top10
country_orders = df['Country'].value_counts().head(10)
axes[1, 0].barh(country_orders.index[::-1], country_orders.values[::-1], color='lightcoral')
axes[1, 0].set_title('订单量Top10国家')
axes[1, 0].set_xlabel('订单数量')

# 5.4 一周各天交易分布（修复：补齐缺失的天）
day_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
day_counts = df['DayOfWeek'].value_counts().sort_index()

# 补齐缺失的天（确保7天都有数据，缺失的填0）
full_day_counts = []
for i in range(7):
    full_day_counts.append(day_counts.get(i, 0))

axes[1, 1].bar(day_names, full_day_counts, color='lightgreen')
axes[1, 1].set_title('一周各天交易分布')
axes[1, 1].set_ylabel('交易数量')

plt.tight_layout()
plt.savefig('data/eda_overview.png', dpi=150, bbox_inches='tight')
print(f" 可视化图表已保存: data/eda_overview.png")
plt.show()

# ==================== 6. 输出关键指标 ====================
print("\n" + "=" * 60)
print("【数据清洗完成 - 关键指标汇总】")
print("=" * 60)
print(f" 总交易记录: {len(df):,}")
print(f" 唯一用户数: {df['CustomerID'].nunique():,}")
print(f" 唯一商品数: {df['StockCode'].nunique():,}")
print(f" 唯一订单数: {df['InvoiceNo'].nunique():,}")
print(f" 总销售额: £{df['Amount'].sum():,.2f}")
print(f" 平均客单价: £{df['Amount'].mean():.2f}")
print(f" 数据时间跨度: {(df['InvoiceDate'].max() - df['InvoiceDate'].min()).days} 天")
print(f" 覆盖国家数: {df['Country'].nunique()}")
print("=" * 60)
print(" Day 1 完成！清洗后的数据已保存，准备进入 Day 2: RFM分析")
print("=" * 60)