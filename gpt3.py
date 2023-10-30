"""Call GPT-3 model to get predictions."""

import dataclasses
import json
import math
import time
from typing import Any
import openai
import argparse

openai.api_key = '<YOUR-GPT-3-KEY>'

RATIONALE_BATCH = 40
TEMPERATURE = 0.7
ENGINE_NAME = 'gpt-3.5-turbo'
INPUT_FILE = '<YOUR-PATH>/coinflip4.tsv'
OUTPUT_PATH = '<YOUR-PATH>/coinflip4_output_sc_001.jsonl'

def arg_parser():
    parser = argparse.ArgumentParser(description="CoT")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--num_postive", type=int, default="2", help="number of postive samples"
    )
    parser.add_argument(
        "--num_negative", type=int, default="2", help="number of negative samples"
    )
    parser.add_argument(
        "--samples", type=int, default="500", help="number of samples to test"
    )
    parser.add_argument(
        "--samples", type=int, default="500", help="number of samples to test"
    )

    return args

args = arg_parser()

@dataclasses.dataclass
class Example:
  question: str
  answer: str
  thought: str


COINFLIP_EXAMPLES = [
    Example(
        question='A coin is heads up. Ka flips the coin. Sherrie flips the coin. Is the coin still heads up?',
        answer='yes',
        thought='The coin was flipped by Ka and Sherrie. So the coin was flipped 2 times, which is an even number. The coin started heads up, so after an even number of flips, it will still be heads up.',
    ),
    Example(
        question='A coin is heads up. Jamey flips the coin. Teressa flips the coin. Is the coin still heads up?',
        answer='yes',
        thought='The coin was flipped by Jamey and Teressa. So the coin was flipped 2 times, which is an even number. The coin started heads up, so after an even number of flips, it will still be heads up.',
    ),
    Example(
        question='A coin is heads up. Maybelle flips the coin. Shalonda does not flip the coin. Is the coin still heads up?',
        answer='no',
        thought='The coin was flipped by Maybelle. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up.',
    ),
    Example(
        question='A coin is heads up. Millicent does not flip the coin. Conception flips the coin. Is the coin still heads up?',
        answer='no',
        thought='The coin was flipped by Conception. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up.',
    ),
    Example(
        question='A coin is heads up. Sal flips the coin. Raymond does not flip the coin. Is the coin still heads up?',
        answer='no',
        thought='The coin was flipped by Sal. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up.',
    ),
    Example(
        question='A coin is heads up. Conception flips the coin. Kristian does not flip the coin. Is the coin still heads up?',
        answer='no',
        thought='The coin was flipped by Conception. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up.',
    ),
    Example(
        question='A coin is heads up. Inga does not flip the coin. Elanor does not flip the coin. Is the coin still heads up?',
        answer='yes',
        thought='The coin was flipped by no one. So the coin was flipped 0 times. The coin started heads up, and it was not flipped, so it is still heads up.',
    ),
    Example(
        question='A coin is heads up. Ryan flips the coin. Shaunda flips the coin. Is the coin still heads up?',
        answer='yes',
        thought='The coin was flipped by Ryan and Shaunda. So the coin was flipped 2 times, which is an even number. The coin started heads up, so after an even number of flips, it will still be heads up.',
    ),
]


with open(INPUT_FILE, 'r') as file:
  lines = file.readlines()
input_list = []
label_list = []

for line in lines:
  fields = line.split('\n')[0].split('\t')
  question = fields[0]
  answer = fields[1]

  for i in range(RATIONALE_BATCH):
    prompt = ''
    for j, ex in enumerate(COINFLIP_EXAMPLES):
      prompt += 'Q: ' + ex.question + '\nA: ' + ex.thought + ' The answer is ' + ex.answer  + '.\n\n'
    input_list.append(prompt + 'Q: ' + question.replace('\\n', '\n') + '\nA:')

print(len(input_list))
print(input_list[0])


def _complete_with_retry(prompt) -> Any:
  try:
    reply = openai.Completion.create(
        engine=ENGINE_NAME,
        prompt=prompt,
        temperature=TEMPERATURE,
        max_tokens=128,
        frequency_penalty=0,
        presence_penalty=0,
        stop=['Q:'])
    return reply

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


start_time = time.time()
BATCH_SIZE = 20
with open(OUTPUT_PATH, 'w') as outfile:
  batches = math.ceil(len(input_list)/BATCH_SIZE)
  print('batches', batches)
  for b in range(batches):
    input_batch = input_list[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
    response = _complete_with_retry(input_batch)
    for input_prompt, res in zip(input_batch, response['choices']):
      pred = {}
      pred['input'] = input_prompt
      pred['output'] = []
      pred['output'].append(res['text'])
      json.dump(pred, outfile)
      outfile.write('\n')

    if b and b % 10 == 0:
      print(b, '/', batches)
      print('time: ', time.time() - start_time)
