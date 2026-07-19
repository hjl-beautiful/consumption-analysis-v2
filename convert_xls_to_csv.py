import pandas as pd

base_path = r'D:\github\consumption-analysis\consumption-analysis\data'

# 转换 rfm_results
df_rfm = pd.read_excel(f'{base_path}\\rfm_results.xls')
df_rfm.to_csv(f'{base_path}\\rfm_results.csv', index=False)
print(f" rfm_results.csv 已生成，{len(df_rfm)} 行")

# 转换 segment_summary
df_seg = pd.read_excel(f'{base_path}\\segment_summary.xls')
df_seg.to_csv(f'{base_path}\\segment_summary.csv', index=False)
print(f" segment_summary.csv 已生成，{len(df_seg)} 行")