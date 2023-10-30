import json
import os
import time
import together
from utils import format_question
import random
import argparse
import requests


def arg_parser():
    parser = argparse.ArgumentParser(description="CoT")
    # ... (Your existing argument configurations)
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
    return input_list, label_list


def _complete_with_retry(prompt, args):
    # ... (Your existing logic for _complete_with_retry)
    return response, done


def main():
    args = arg_parser()
    initialize_environment(args)
    
    output_path = os.path.join(args.data_path, args.save_path)

    COINFLIP_EXAMPLES = load_dataset('dataset/CSQA/train_rand_split.jsonl')
    lines = load_dataset(args.dataset_path)

    input_list, label_list = build_input_and_labels(lines, COINFLIP_EXAMPLES, args)

    start_time = time.time()
    with open(output_path, 'w') as outfile:
        for index in range(args.samples):
            # ... (Your existing logic for loop body)
            print(index)
            if index and index % 50 == 0:
                print(index, '/', len(input_list))
                print('time: {:.2f}'.format(time.time() - start_time))


if __name__ == "__main__":
    main()
