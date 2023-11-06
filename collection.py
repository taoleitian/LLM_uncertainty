from collections import Counter
import json
import math

# Assuming 'output' is a list of strings containing the answers and confidences
with open('results/calibation/calibation.jsonl', 'r') as f:
    lines = f.readlines()
    # Initialize a Counter to store the counts of each answer
    answer_counts = Counter()
    for line in lines:
        output = json.loads(line)["output"]
        # Iterate through each string in the output list
        for item in output:
            # Extract the answer which is the first character of the string
            answer = item.strip()[0]
            # Check if the answer is a valid option before counting
            if answer in ['A', 'B', 'C', 'D', 'E']:
                answer_counts[answer] += 1
        total_answers = sum(answer_counts.values())

        # 计算每个答案的概率
        answer_probabilities = [count / total_answers for count in answer_counts.values()]

        # 计算熵
        entropy = -sum(p * math.log(p, 2) for p in answer_probabilities if p > 0)
        max_entropy = math.log2(len(answer_counts))

        # 如果所有答案都是可能的，即使没有在answer_counts中出现，我们需要包括它们
        # max_entropy = math.log2(number_of_possible_answers)

        # 标准化熵：将熵的值从[0, max_entropy]映射到[0, 1]
        normalized_entropy = entropy / max_entropy

        # 计算置信度：一个从不确定性到确定性的映射
        confidence = (1 - normalized_entropy)*4

        print(f"The confidence level based on the entropy is: {confidence}")
