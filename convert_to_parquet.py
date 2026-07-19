import pandas as pd

# 读取原 CSV
df = pd.read_csv(r'D:\github\consumption-analysis\consumption-analysis\data\OnlineRetail_cleaned.csv')

# 保存为 parquet
df.to_parquet(r'D:\github\consumption-analysis\consumption-analysis\data\OnlineRetail_cleaned.parquet', index=False)

print(f" 转换完成！")
print(f"原 CSV 行数: {len(df)}")
print(f"列数: {len(df.columns)}")