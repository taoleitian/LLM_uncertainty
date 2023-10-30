import json
import utils
import numpy as np

with open('results/CSQA/with_negtive_30.jsonl', 'r') as f:
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
  #maj_ans = utils.vote_based_on_confidence(ans_list, confidence_list, 0.1)
  target = datas['answer']
  if str(target) == str(maj_ans):
    correct += 1
  confi_list.append(np.mean(confidence_list))
  pred_list.append(maj_ans)
  gt_list.append(target)


# 随机生成置信度作为示例

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

# 使用梯形法则计算AUC
auc = np.trapz(selective_accuracies, coverages)
print("AUC:", auc)

total = len(lines)
print(correct, total, correct/total)