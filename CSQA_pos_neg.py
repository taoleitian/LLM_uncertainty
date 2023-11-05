import os
import time
from typing import Any
import together
from utils import format_question
import random
import argparse
import requests  
import json
random.seed(0)

def arg_parser():
    parser = argparse.ArgumentParser(description="CoT")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--num_positive", type=int, default="2", help="number of postive samples"
    )
    parser.add_argument(
        "--num_negative", type=int, default="2", help="number of negative samples"
    )
    parser.add_argument(
        "--samples", type=int, default="1221", help="number of samples to test"
    )
    parser.add_argument(
        "--SC_times", type=int, default="1", help="number of SC_times"
    )
    parser.add_argument(
        "--save_path", type=str, default="save_path.jsonl", help="api keys"
    )
    parser.add_argument(
        "--data_path", type=str, default="results/pos_neg", help="api keys"
    )
    parser.add_argument(
        "--temprature", type=float, default="0.7", help="api keys"
    )
    parser.add_argument(
        "--model_type", type=str, default="togethercomputer/LLaMA-2-7B-32K", help="api keys"
    )  
    parser.add_argument(
        "--max_tokens", type=int, default="50", help="api keys"
    ) 
    parser.add_argument(
        "--api_keys", type=str, default="98e058c881079af8221b192917287b3d82856fafb9d066e8828754a49c9f60ee", help="api keys"
    ) 
    parser.add_argument(
        "--dataset_path", type=str, default='dataset/CSQA/dev_rand_split.jsonl', help="api keys"
    )

    args = parser.parse_args()
    return args


def initialize_environment(args):
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)
    together.api_key = args.api_keys


def load_dataset(dataset_path):
    with open(dataset_path, 'r') as file:
        lines = file.readlines()
    return lines


def build_input_and_labels(lines, COINFLIP_EXAMPLES, args):
    input_list = []
    label_list = []
    # ... (Your existing logic to build input_list and label_list)
      # Postive examples
    for line in lines:
        fields = json.loads(line.split('\n')[0].split('\t')[0])
        question = format_question(fields, is_val=False)
        answer = fields["answerKey"]

        prompt = 'Answer the following question to the best of your ability, and provide a score between 0 and 1 to indicate the confidence you have in your answer. Confidence scores closer to 0 indicate you have less confidence in your answer, while scores closer to 1 indicate you have more confidence in your answer. You must answer the question with one of the valid choices. \n\n'
        '''
        if args.num_positive > 0:
            prompt += "Here are some positive samples. Since the answer is correct, the confidence level is extremely high, close to 1.\n"
        '''
        for index in range(args.num_positive):
            #prompt += "This is a positive sample. As the answer is correct, the confidence is high, close to 1.\n"
            example = json.loads(COINFLIP_EXAMPLES[index])
            question_prompt = format_question(example, is_val=True, if_confidence=True)
            prompt += question_prompt #+ '\n'+'Confidence: ' + str(round(random.uniform(0.8, 1.0), 2)) + '.\n\n'
            #prompt += question_prompt + '\n'+'Confidence: ' + str(round(random.uniform(0.8, 1.0), 2)) + '.\n\n'
        '''
        if args.num_negative > 0:         
        # Negtive examples  
            prompt += "Here are some negtive samples. Since the answer is wrong, the confidence level is extremely low, close to 0.\n"
        '''
        for index in range(args.num_positive, args.num_positive + args.num_negative):
            prompt += "This is a negative sample. As the answer is wrong, the confidence is low, close to 0.\n"
            example = json.loads(COINFLIP_EXAMPLES[index])
            question_prompt = format_question(example, is_val=True,answer_True=False)
            prompt += question_prompt + '\n'+'Confidence: ' + str(round(random.uniform(0.0, 0.2), 2)) + '.\n\n'
            #print(prompt)


        input_list.append(prompt + question)
        label_list.append(answer)
    print(len(input_list))
    print(input_list[0])
    return input_list, label_list


def _complete_with_retry(args, prompt) -> Any:
    response = None 
    done = False
    try:
        response = together.Complete.create(
            model="togethercomputer/LLaMA-2-7B-32K",
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temprature,  # 假设TEMPERATURE是0.7
            top_k=50,
            top_p=0.7,
            repetition_penalty=0,
            stop="Question:"
        )
        done = True
        return response, done
    except requests.exceptions.HTTPError as e:
        print('======> Service unavailable error: will retry after 30 seconds')
        time.sleep(30)
        return _complete_with_retry(args,prompt)

    except Exception as e:  # pylint: disable=broad-except
        print(f'Exception: {e}')
        return '', done


def main():
    args = arg_parser()
    initialize_environment(args)
    
    output_path = os.path.join(args.data_path, args.save_path)

    COINFLIP_EXAMPLES = load_dataset('dataset/CSQA/test_rand_split_no_answers.jsonl')
    lines = load_dataset(args.dataset_path)

    input_list, label_list = build_input_and_labels(lines, COINFLIP_EXAMPLES, args)
    print(len(input_list))
    start_time = time.time()
    with open(output_path, 'w') as outfile:
        for index in range(args.samples):
            input_batch = input_list[index]
            pred = {}
            pred['input'] = input_batch
            pred['output'] = []
            for time_prompt in range(args.SC_times):
                response, done = _complete_with_retry(args, input_batch)
                if done:
                    pred['output'].append(response['output']['choices'][0]['text'])
            pred['answer'] = label_list[index]
            json.dump(pred, outfile)
            outfile.write('\n')
            #print(index)
            if index and index % 50 == 0:
                print(index, '/', len(input_list))
                print('time: {:.2f}'.format(time.time() - start_time))


if __name__ == "__main__":
    main()
