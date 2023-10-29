import json
import utils

with open('<file_path>', 'r') as f:
  lines = f.readlines()

correct = 0
for line in lines:
  data = json.loads(line)
  pred_list = []
  for pred in data['output']:
    ans = utils.get_ans(pred)
    if ans:
      pred_list.append(ans)
  if not pred_list:
    continue
  maj_ans = utils.get_maj(pred_list)
  target = data['answer']
  if utils._is_float(target) and utils._is_float(maj_ans):
    if abs(float(target) - float(maj_ans)) <= 1e-5:
      correct += 1
  elif str(target) == str(maj_ans):
    correct += 1

total = len(lines)
print(correct, total, correct/total)