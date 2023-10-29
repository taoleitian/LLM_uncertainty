import re
from collections import defaultdict

NUMBER_SET = [str(num) for num in range(0, 10)]

def _is_float(s):
  try:
    float(s)
    return True
  except:
    return False

FINAL_ANS = 'Answer:'
FINAL_CONFIDENCE = 'Confidence:'

def format_question(json_data, is_val=False):
    question_concept = json_data.get("question_concept", "Unknown")
    stem = json_data.get("question", {}).get("stem", "Unknown question")
    choices = json_data.get("question", {}).get("choices", [])
    answer_key = json_data.get("answerKey", "Unknown")

    formatted_str = f"Question: {stem}\n"
    
    formatted_str += f"Question Concept: {question_concept}\n"
    formatted_str += f"Let us think it step by step.\n"
    
    for choice in choices:
        label = choice.get("label", "Unknown label")
        text = choice.get("text", "Unknown text")
        formatted_str += f"{label}. {text}\n"
    if is_val ==True:    
      formatted_str += f"Answer: {answer_key}"
    else:
      formatted_str += f"Answer:"

    return formatted_str


def clean_ans(ans):
  index = ans.find('.')
  if index >= 0:
    end_index = index + 1
    while end_index < len(ans) and ans[end_index] in NUMBER_SET:
      end_index += 1
    ans = ans[:end_index]
  while ans and ans.endswith('.'):
    ans = ans[:-1]
  
  ans = ans.split('=')[-1].strip()
  for c in ['$', ',', '%', '€', '"']:
    ans = ans.replace(c, '')
  parts = ans.split(' ')
  for part in parts:
    if _is_float(part):
      return part
  
  ans = parts[0]  # default
  for part in parts:
    if not part.isalpha():  # take the 1st non-alpha token
      ans = part
      break
  while ans and ans[-1].isalpha():
    ans = ans[:-1]
  return ans.strip()

def get_ans(pred):
  text = pred.split('Question:')[0].split('[eot]')[0].replace('\n', '').strip()
  if text.rfind(FINAL_ANS) >= 0:
    pred_ans = text[text.rfind(FINAL_ANS) + len(FINAL_ANS):len(text)].strip()
    return clean_ans(pred_ans)
  else:
    return ''


from collections import Counter
def get_maj(ans_list):
  is_all_float = True
  float_list = []
  for ans in ans_list:
    if _is_float(ans):
      float_list.append(float(ans))
    else:
      is_all_float = False
      break
  if is_all_float:
    f = Counter(float_list)
    return f.most_common()[0][0]
  else:
    c = Counter(ans_list)
    return c.most_common()[0][0]

def get_str_ans(pred):
  text = pred.split('Question:')[0].split('[eot]')[0].replace('\n', '').strip()
  if text.rfind(FINAL_ANS) >= 0:
    pred_ans = text[text.rfind(FINAL_ANS) + len(FINAL_ANS):len(text)].strip()
    if pred_ans.endswith('.'):
      pred_ans = pred_ans[:-1]
    if pred_ans.rfind("$") >= 0:
      pred_ans = pred_ans[pred_ans.rfind("$")+1:]
    return pred_ans
  else:
    return ''
  
def get_ans_choice_confidence(pred):
  text = pred.split('Q')[0].split('[eot]')[0].replace('\n', '').strip()
  if text.rfind(FINAL_CONFIDENCE) >= 0:
    confidence = text[text.rfind(FINAL_CONFIDENCE) + len(FINAL_CONFIDENCE):len(text)].strip()
    if confidence.endswith('.'):
      confidence = confidence[:-1]

  else:
    return ''
  pred_ans = text[0]
  return pred_ans, float(confidence)

import numpy as np
def get_str_ans_confidence(pred):
  text = pred.split('Q:')[0].split('[eot]')[0].replace('\n', '').strip()
  if text.rfind(FINAL_CONFIDENCE) >= 0:
    confidence = text[text.rfind(FINAL_CONFIDENCE) + len(FINAL_CONFIDENCE):len(text)].strip()
    if confidence.endswith('.'):
      confidence = confidence[:-1]

  else:
    return ''
  if text.rfind(FINAL_ANS) >= 0:
    pred_ans = text[text.rfind(FINAL_ANS) + len(FINAL_ANS):len(text)].strip()
    if pred_ans.endswith('.'):
      pred_ans = pred_ans[:-1]
    if pred_ans.rfind("$") >= 0:
      pred_ans = pred_ans[pred_ans.rfind("$")+1:]
    return pred_ans, float(confidence)
  else:
    return ''
def softmax_with_temperature(confidences, temperature=1.0):
    """
    使用temperature参数执行softmax归一化
    :param confidences: 包含confidence值的列表
    :param temperature: temperature参数控制softmax的“软硬度”
    :return: softmax归一化后的confidence值列表
    """
    # 首先，根据temperature调整confidences
    adjusted_confidences = np.array(confidences) / temperature
    
    # 计算每个调整后的confidence的指数
    exp_confidences = np.exp(adjusted_confidences)
    
    # 计算指数的总和
    sum_exp_confidences = np.sum(exp_confidences)
    
    # 对每个confidence的指数进行归一化
    softmax_confidences = exp_confidences / sum_exp_confidences
    
    return softmax_confidences.tolist()
def vote_based_on_confidence(answers, confidences, temp=1.0):
    votes = defaultdict(float)
    confidences = softmax_with_temperature(confidences, temperature=temp)
    for answer, confidence in zip(answers, confidences):
        # 将confidence score作为票数
        votes[answer] += confidence
        
    # 根据票数对答案进行排序
    sorted_votes = sorted(votes.items(), key=lambda item: item[1], reverse=True)
    
    # 返回获得最多票数的答案
    return sorted_votes[0][0]