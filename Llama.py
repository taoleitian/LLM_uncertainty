"""Call GPT-3 model to get predictions."""

import dataclasses
import json
import math
import time
from typing import Any
import together


together.api_key='98e058c881079af8221b192917287b3d82856fafb9d066e8828754a49c9f60ee'
model_list = together.Models.list()
print(f"{len(model_list)} models available")

RATIONALE_BATCH = 1
TEMPERATURE = 0.7
ENGINE_NAME = 'gpt-3.5-turbo'
INPUT_FILE = 'dataset/GSM8K/test.jsonl'
OUTPUT_PATH = 'results/GSM8K_SELF_CONSISTENCY_20_400.jsonl'


@dataclasses.dataclass
class Example:
  question: str
  answer: str
  thought: str
  confidence: str



COINFLIP_EXAMPLES = [
    Example(
        question='Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?',
        answer='72',
        confidence='0.86',
        thought='Natalia sold 48 clips in April and then she sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May.',
    ),

    Example(
        question='Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?',
        answer='10',
        confidence='0.73',
        thought='Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $10.',
    ),

    Example(
        question='Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?',
        answer='5',
        confidence='0.10',
        thought='In the beginning, Betty has only 100 / 2 = $50. Betty’s grandparents gave her 15 * 2 = $30. This means, Betty needs 100 - 50 - 30 - 15 = $5 more.',
    ),

]
COINFLIP_EXAMPLES = COINFLIP_EXAMPLES[:1 ]
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
      prompt += 'Q: ' + ex.question + '\nA: ' + ex.thought + ' The answer is ' + ex.answer  + '.\nC: ' + ex.confidence +  '.\n\n'
    input_list.append(prompt + 'Q: ' + question.replace('\\n', '\n') + '\nA:')
    label_list.append(answer.split('\n#### ')[-1])

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
        model ='togethercomputer/Llama-2-7B-32K-Instruct',
        prompt = prompt, 
        #prompt = prompt,  
        max_tokens = 512,
        temperature = 0.8,
        top_k = 50,
        top_p = 0.7,
        repetition_penalty = 1.1,
        stop = "Q:"
    )
    return response

start_time = time.time()
SELF_CONSISTENCY = 30
with open(OUTPUT_PATH, 'w') as outfile:
  #batches = math.ceil(len(input_list)/BATCH_SIZE)
  #print('batches', batches)
  #num_examples = len(input_list)
  #num_examples = len(input_list)
  num_examples = 400
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
