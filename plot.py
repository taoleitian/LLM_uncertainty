import csv
import matplotlib.pyplot as plt

# 从CSV文件中读取数据
num_positives = []
accs = []
eces = []
aucs = []

with open('10_8_shot.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        num_positives.append(int(row['num_negative']))
        accs.append(float(row['acc']))
        eces.append(float(row['ece']))
        aucs.append(float(row['auc']))

# 根据num_positive排序数据
sorted_data = sorted(zip(num_positives, accs, eces, aucs))
num_positives, accs, eces, aucs = zip(*sorted_data)

# 创建折线图
plt.figure(figsize=(10,6))
plt.plot(num_positives, accs, marker='o', label='Accuracy (acc)')
plt.plot(num_positives, eces, marker='o', label='ECE (ece)')
plt.plot(num_positives, aucs, marker='o', label='AUC (auc)')

# 设置标题和标签
plt.title("Results vs. Number of Positives")
plt.xlabel("Number of Positives (num_positive)")
plt.ylabel("Values")
plt.legend()
plt.grid(True)

# 显示图形
plt.show()
