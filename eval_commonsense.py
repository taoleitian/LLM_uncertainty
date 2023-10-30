import json
import utils
import numpy as np
import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description="CoT")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--save_path", type=str, default=".csv", help="number of postive samples"
    )
    parser.add_argument(
        "--results_path", type=str, default=".csv", help="number of postive samples"
    )

    args = parser.parse_args()
    return args

args = arg_parser()

with open('results/two_type_shot/two.jsonl', 'r') as f:
  lines = f.readlines()

correct = 0
pred_list = []
gt_list = []
confi_list = []
for line in lines:
  datas = json.loads(line)
  ans_list = []
  confidence_list = []
  for index in range(len(datas['output'])):
  #for index in range(2):
    pred = datas['output'][index]
    #print(index)
  #for pred in data['output'][0][0]['text']:
    output = utils.get_ans_choice_confidence(pred)
    if output:
      ans, confidence = output
      ans_list.append(ans)
      confidence_list.append(confidence)
  if not pred:
    continue
  maj_ans = utils.get_maj(ans_list)
  #maj_ans = utils.vote_based_on_confidence(ans_list, confidence_list, 100)
  #maj_ans = utils.major_vote_top_k(ans_list, confidence_list, 30)
  target = datas['answer']
  if str(target) == str(maj_ans):
    correct += 1
  confi_list.append(np.mean(confidence_list))
  pred_list.append(maj_ans)
  gt_list.append(target)


# 随机生成置信度作为示例
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
# 根据置信度排序数据
sorted_indices = np.argsort(confi_list)[::-1]
y_pred_sorted = np.array(pred_list)[sorted_indices]
y_true_sorted = np.array(gt_list)[sorted_indices]

# 计算选择性准确率A(c)对于每个可能的c
coverages = np.linspace(0, 1, 20)[1:]  # [0.25, 0.5, 0.75, 1.0]
selective_accuracies = []

for c in coverages:
    top_n = int(len(pred_list) * c)
    top_n_predictions = y_pred_sorted[:top_n]
    top_n_true = y_true_sorted[:top_n]
    
    accuracy_at_c = np.mean(top_n_predictions == top_n_true)
    selective_accuracies.append(accuracy_at_c)
compute_ece(pred_list, gt_list, confi_list)
# 使用梯形法则计算AUC
auc = np.trapz(selective_accuracies, coverages)
print("AUC:", auc)

total = len(lines)
print(correct, total, correct/total)