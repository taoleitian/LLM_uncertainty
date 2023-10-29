import json
import utils

with open('results/GSM8K_SELF_CONSISTENCY_20_400.jsonl', 'r') as f:
  lines = f.readlines()

correct = 0
for line in lines:
  datas = json.loads(line)
  ans_list = []
  confidence_list = []
  for index in range(len(datas['output'])):
  #for index in range(2):
    pred = datas['output'][index]
    print(index)
  #for pred in data['output'][0][0]['text']:
    output = utils.get_str_ans_confidence(pred)
    if output:
      ans, confidence = output
      ans_list.append(ans)
      confidence_list.append(confidence)
  if not pred:
    continue
  maj_ans = utils.get_maj(ans_list)
  #maj_ans = utils.vote_based_on_confidence(ans_list, confidence_list, 20)
  maj_ans = ans
  target = datas['answer']
  if str(target) == str(maj_ans):
    correct += 1

total = len(lines)
print(correct, total, correct/total)