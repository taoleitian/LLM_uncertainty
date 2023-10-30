import numpy as np

def compute_ece(answer, output, confidence, num_bins=10):
    # 初始化变量
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_acc = np.zeros(num_bins)
    bin_conf = np.zeros(num_bins)
    bin_count = np.zeros(num_bins)
    
    n = len(answer)
    
    # 创建一个数组来存储每个样本是否被正确分类（1 = 正确，0 = 错误）
    correct = np.array([1 if a == o else 0 for a, o in zip(answer, output)])
    
    # 对每个样本进行统计
    for c, conf in zip(correct, confidence):
        # 查找对应的 bin
        bin_idx = np.digitize(conf, bin_boundaries) - 1
        bin_idx = np.clip(bin_idx, 0, num_bins - 1)
        
        # 更新 bin 统计
        bin_count[bin_idx] += 1
        bin_acc[bin_idx] += c
        bin_conf[bin_idx] += conf
    
    # 计算每个 bin 的准确率和置信度
    bin_acc /= (bin_count + 1e-15)
    bin_conf /= (bin_count + 1e-15)
    
    # 计算 ECE
    ece = np.sum(bin_count / n * np.abs(bin_acc - bin_conf))
    
    return ece

# 真实标签
answer = ['A', 'B', 'A', 'B', 'D']

# 模型预测
output = ['C', 'B', 'C', 'D', 'A']

# 模型预测的置信度
confidence = [0.8, 0.9, 0.7, 0.6, 0.5]

# 计算 ECE
ece = compute_ece(answer, output, confidence)
print(f"Expected Calibration Error (ECE): {ece:.4f}")
