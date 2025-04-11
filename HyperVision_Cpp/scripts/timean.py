import pandas as pd
import matplotlib.pyplot as plt

# 加载数据时指定数值列
data = pd.read_csv('/data16/yurun/HyperVision_Cpp/results/module_timing_details.csv')  # 替换为实际路径

# 确保所有时间列是数值类型
time_columns = ['data_preparation', 'flow_construct', 'edge_construct', 
                'graph_parse', 'graph_detect', 'final_score_calc']
for col in time_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')  # 强制转换为数值

# 绘制趋势图
plt.figure(figsize=(12, 6))
for column in time_columns:
    plt.plot(data['Window'], data[column], label=column, marker='o')

plt.title('Module Execution Time Trend Across Windows')
plt.xlabel('Window')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True)
plt.savefig('/data16/yurun/HyperVision_Cpp/scripts/time_analysis.png')  # 保存到文件
plt.close()

modules = data.columns[1:-2]
bottom = None
plt.figure(figsize=(10, 6))

for module in modules:
    if module != 'incremental_learning':
        plt.bar(data['Window'], data[module], label=module, bottom=bottom)
        if bottom is None:
            bottom = data[module]
        else:
            bottom += data[module]

plt.title('Time Distribution by Module')
plt.xlabel('Window')
plt.ylabel('Total Time (seconds)')
plt.legend()
plt.savefig('/data16/yurun/HyperVision_Cpp/scripts/time_distribution.png')  # 保存到文件
plt.close()

from math import pi

stats = pd.DataFrame({
    'Module': ['data_preparation', 'flow_construct', 'edge_construct', 
               'graph_parse', 'graph_detect'],
    'Mean': [0.705833, 2.21783, 2.356, 0.129667, 4.94808]
})

categories = stats['Module']
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

plt.xticks(angles[:-1], categories)
ax.plot(angles, list(stats['Mean']) + [stats['Mean'][0]], linewidth=2, linestyle='solid')
ax.fill(angles, list(stats['Mean']) + [stats['Mean'][0]], alpha=0.25)

plt.title('Mean Execution Time by Module (Radar Chart)')
plt.savefig('/data16/yurun/HyperVision_Cpp/scripts/radar_chart.png')  # 保存到文件
plt.close()