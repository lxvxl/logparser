import pandas as pd
import sys

# 从命令行获取参数
csv_file_path = sys.argv[1]
column_name = sys.argv[2]

# 读取csv文件
df = pd.read_csv(csv_file_path)

# 按照指定列名排序
df.sort_values(by=column_name, inplace=True)

# 将排序后的结果写回源文件
df.to_csv(csv_file_path, index=False)