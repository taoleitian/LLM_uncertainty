"""Call GPT-3 model to get predictions."""

import dataclasses
import json
import math
import time
from typing import Any
import together
from utils import format_question
import random
random.seed(0)
together.api_key='98e058c881079af8221b192917287b3d82856fafb9d066e8828754a49c9f60ee'
model_list = together.Models.list()
print(f"{len(model_list)} models available")

RATIONALE_BATCH = 1
TEMPERATURE = 0.7
ENGINE_NAME = 'gpt-3.5-turbo'
INPUT_FILE = 'dataset/CSQA/dev_rand_split.jsonl'
OUTPUT_PATH = 'results/CSQA/baseline.jsonl'



COINFLIP_EXAMPLES = []
FEW_SHOT = 1
formatted_questions = []
with open('dataset/CSQA/train_rand_split.jsonl', 'r') as file:
  few_shot_lines = file.readlines()
for index in range(FEW_SHOT):
    json_list = json.loads(few_shot_lines[index].split('\n')[0].split('\t')[0])
    formatted_str = format_question(json_list, is_val=True)
    COINFLIP_EXAMPLES.append(formatted_str)

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
    for j, ex in enumerate(COINFLIP_EXAMPLES):
      prompt += ex + '\n'+'Confidence: ' + str(round(random.uniform(0.8, 1.0), 2)) + '.\n\n'
    input_list.append(prompt + 'Q: ' + question)
    label_list.append(answer)

print(len(input_list))
print(input_list[0])

'''
def _complete_with_retry(prompt) -> Any:
  try:
    
    url = "https://api.together.xyz/inference"
    payload = {
        "model": "togethercomputer/llama-2-13b",
        "prompt": [prompt, prompt],
        "max_tokens": 128,
        "stop": "Q:",
        "temperature": 0,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1
    }
    headers = {
        #"accept": "application/json",
        #"content-type": "application/json",
        "Authorization": "Bearer 98e058c881079af8221b192917287b3d82856fafb9d066e8828754a49c9f60ee"
    }

    response = requests.post(url, json=payload, headers=headers)

    #print(response.text)
    response = json.loads(response.text)['output']
    return response

  except openai.error.RateLimitError:
    print('======> Rate limit error')
    time.sleep(10)
    return _complete_with_retry(prompt)
  except openai.error.ServiceUnavailableError:
    print('======> Service unavailable error: will retry after 60 seconds')
    time.sleep(30)
    return _complete_with_retry(prompt)
  except Exception:  # pylint: disable=broad-except
    print('Exception!!')
    return ''
'''

def _complete_with_retry(prompt) -> Any:
    #prompt = [prompt]
    response = together.Complete.create(
        model='togethercomputer/Llama-2-7B-32K-Instruct',
        prompt=prompt, 
        #prompt = prompt,  
        max_tokens=30,
        temperature=TEMPERATURE,
        top_k=50,
        top_p=0.7,
        repetition_penalty=0,
        stop="Q:"
    )
    return response

start_time = time.time()
SELF_CONSISTENCY = 30
with open(OUTPUT_PATH, 'w') as outfile:

  num_examples = 300
  for index in range(num_examples):
    input_batch = input_list[index]
    pred = {}
    pred['input'] = input_batch
    pred['output'] = []
    for time_prompt in range(SELF_CONSISTENCY):
      response = _complete_with_retry(input_batch)
      pred['output'].append(response['output']['choices'][0]['text'])
    pred['answer'] = label_list[index]
    json.dump(pred, outfile)
    outfile.write('\n')
    print(index)

    if index and index % 10 == 0:
      print(index, '/', len(input_list))
      print('time: ', time.time() - start_time)
