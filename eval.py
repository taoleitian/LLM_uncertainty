import json
import numpy as np
import argparse
import utils

def arg_parser():
    parser = argparse.ArgumentParser(description="CoT")
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--data_path", type=str, default="results/tlt/positive.jsonl")
    parser.add_argument("--results_path", type=str, default=".csv")
    args = parser.parse_args()
    return args

def load_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines

def process_data(lines):
    pred_list, gt_list, confi_list = [], [], []
    for line in lines:
        datas = json.loads(line)
        ans_list, confidence_list = [], []
        for pred in datas['output'][:1]:
            output = utils.get_ans_choice_confidence(pred)
            if output:
                ans, confidence = output
                ans_list.append(ans)
                confidence_list.append(confidence)
        maj_ans = utils.get_maj(ans_list) if ans_list else None
        if confidence_list:
            confi_list.append(np.mean(confidence_list))
            pred_list.append(maj_ans)
            gt_list.append(datas['answer'])
    return pred_list, gt_list, confi_list

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

def compute_acc(pred_list, gt_list):
    correct = np.sum(np.array(pred_list) == np.array(gt_list))
    total = len(gt_list)
    return correct / total
def get_results(pred_list, gt_list, confi_list):
    ece = compute_ece(pred_list, gt_list, confi_list)
    print(f"Expected Calibration Error (ECE): {ece:.4f}")

    acc = compute_acc(pred_list, gt_list)
    print(f"Accuracy: {acc:.4f}")

    auc = compute_auc_and_coverage(pred_list, gt_list, confi_list)
    print(f"AUC: {auc:.4f}")
    return acc, ece, auc

def compute_auc_and_coverage(pred_list, gt_list, confi_list):
    sorted_indices = np.argsort(confi_list)[::-1]
    y_pred_sorted = np.array(pred_list)[sorted_indices]
    y_true_sorted = np.array(gt_list)[sorted_indices]
    coverages = np.linspace(0, 1, 10)[1:]
    selective_accuracies = []
    for c in coverages:
        top_n = int(len(pred_list) * c)
        top_n_predictions = y_pred_sorted[:top_n]
        top_n_true = y_true_sorted[:top_n]
        accuracy_at_c = np.mean(top_n_predictions == top_n_true)
        selective_accuracies.append(accuracy_at_c)
    auc = np.trapz(selective_accuracies, coverages)
    return auc

def main():
    args = arg_parser()
    data_path = args.data_path
    lines = load_data(data_path)
    pred_list, gt_list, confi_list = process_data(lines)
    acc, ece, auc = get_results(pred_list, gt_list, confi_list)

if __name__ == "__main__":
    main()
