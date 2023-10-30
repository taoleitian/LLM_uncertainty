import csv
import os
from eval import get_results, load_data, process_data

# 创建一个新的CSV文件并写入列名
with open('results.csv', 'w', newline='') as csvfile:
    fieldnames = ['num_positive', 'num_negative', 'temprature', 'counters', 'acc', 'ece', 'auc']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    temprature = 0.7
    counters = range(1, 6)
    combinations = [(i, 8 - i) for i in range(9)]

    for num_positive, num_negative in combinations:
        for counter in counters:
            save_path = f"{num_positive}_{num_negative}_{int(temprature * 10)}_{str(counter).zfill(2)}.jsonl"
            cmd = f"python3 CSQA.py --num_positive {num_positive} --num_negative {num_negative} --temprature {temprature} --save_path {save_path}"
            print(cmd)
            os.system(cmd)
            os.path.join("results/pos_neg", save_path)
            lines = load_data(save_path)
            pred_list, gt_list, confi_list = process_data(lines)
            acc, ece, auc = get_results(pred_list, gt_list, confi_list)

            # 将结果写入CSV文件
            writer.writerow({'num_positive': num_positive, 'num_negative': num_negative, 'temprature': temprature, 'counters': counter, 'acc': acc, 'ece': ece, 'auc': auc})


        

