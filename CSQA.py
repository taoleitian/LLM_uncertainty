import json
import math
import time
from typing import Any
import together
from utils import format_question
import random
random.seed(0)


together.api_key='98e058c881079af8221b192917287b3d82856fafb9d066e8828754a49c9f60ee'

#together.api_key='54dd8b1db669c11142c88e43391741d5cd5865078c9a95095a4db72ba89b6bbc'
model_list = together.Models.list()
print(f"{len(model_list)} models available")

RATIONALE_BATCH = 1
TEMPERATURE = 0.7
ENGINE_NAME = 'gpt-3.5-turbo'
INPUT_FILE = 'dataset/CSQA/dev_rand_split.jsonl'
OUTPUT_PATH = 'results/CSQA/SC_30_500.jsonl'



COINFLIP_EXAMPLES = []
FEW_SHOT = 4
formatted_questions = []
with open('dataset/CSQA/train_rand_split.jsonl', 'r') as file:
  few_shot_lines = file.readlines()
for index in range(FEW_SHOT):
    json_list = json.loads(few_shot_lines[index].split('\n')[0].split('\t')[0])
    #formatted_str = format_question(json_list, is_val=True)
    COINFLIP_EXAMPLES.append(json_list)

print(COINFLIP_EXAMPLES)
with open(INPUT_FILE, 'r') as file:
  lines = file.readlines()
input_list = []
label_list = []

for line in lines:
  fields = json.loads(line.split('\n')[0].split('\t')[0])
  question = format_question(fields, is_val=False)
  answer = fields["answerKey"]

  for i in range(RATIONALE_BATCH):
    prompt = 'Answer the following question to the best of your ability, and provide a score between 0 and 1 to indicate the confidence you have in your answer. Confidence scores closer to 0 indicate you have less confidence in your answer, while scores closer to 1 indicate you have more confidence in your answer. You must answer the question with one of the valid choices. \n\n'

    # Postive examples
    prompt += "Here are some positive samples. Since the answer is correct, the confidence level is extremely high, close to 1.\n"
    for index in range(len(COINFLIP_EXAMPLES)//2):
      question_prompt = format_question(COINFLIP_EXAMPLES[index], is_val=True)
      prompt += question_prompt + '\n'+'Confidence: ' + str(round(random.uniform(0.8, 1.0), 2)) + '.\n\n'
      #print(prompt)

    # Negtive examples
    prompt += "Here are some negtive samples. Since the answer is wrong, the confidence level is extremely high, close to 0.\n"
    for index in range(len(COINFLIP_EXAMPLES)//2, len(COINFLIP_EXAMPLES)):
      question_prompt = format_question(COINFLIP_EXAMPLES[index], is_val=True,answer_True=False)
      prompt += question_prompt + '\n'+'Confidence: ' + str(round(random.uniform(0.0, 0.2), 2)) + '.\n\n'
      #print(prompt)
    input_list.append(prompt + question)
    label_list.append(answer)

print(len(input_list))
print(input_list[0])


def _complete_with_retry(prompt) -> Any:
    response = None  # 初始化response为None
    done = False
    try:
        response = together.Complete.create(
            model='togethercomputer/LLaMA-2-7B-32K',
            prompt=prompt,
            max_tokens=30,
            temperature=0.7,  # 假设TEMPERATURE是0.7
            top_k=50,
            top_p=0.7,
            repetition_penalty=0,
            stop="Question:"
        )
        done = True
        return response, done
    except requests.exceptions.HTTPError as e:
        if response and response.status_code > 500:  # 确保response不是None
            print('======> Service unavailable error: will retry after 30 seconds')
            time.sleep(30)
            return _complete_with_retry(prompt)
        else:
            print(f"HTTP Error: {e}")
            return '', done
    except Exception as e:  # pylint: disable=broad-except
        print(f'Exception: {e}')
        return '', done


def _complete_with_retry_s(prompt) -> Any:
    done = False
    response = together.Complete.create(
            model='togethercomputer/LLaMA-2-7B-32K',
            prompt=prompt, 
            #prompt = prompt,  
            max_tokens=30,
            temperature=TEMPERATURE,
            top_k=50,
            top_p=0.7,
            repetition_penalty=0,
            stop="Question:"
    )
    done = True
    return response, done


start_time = time.time()
SELF_CONSISTENCY = 1
with open(OUTPUT_PATH, 'w') as outfile:

  num_examples = 500
  for index in range(num_examples):
    input_batch = input_list[index]
    pred = {}
    pred['input'] = input_batch
    pred['output'] = []
    for time_prompt in range(SELF_CONSISTENCY):
      response, done = _complete_with_retry(input_batch)
      if done:
        pred['output'].append(response['output']['choices'][0]['text'])
    pred['answer'] = label_list[index]
    json.dump(pred, outfile)
    outfile.write('\n')
    print(index)

    if index and index % 10 == 0:
      print(index, '/', len(input_list))
      print('time: ', time.time() - start_time)
