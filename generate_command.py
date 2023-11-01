import csv
import os
import concurrent.futures
import queue
from eval import get_results, load_data, process_data
print([(2 * i, 8 - 2 * i) for i in range(5)])
exit()
def process_combination(num_positive, num_negative, counter, temprature, result_queue):
    save_path = f"{num_positive}_{num_negative}_{int(temprature * 10)}_{str(counter).zfill(2)}.jsonl"
    cmd = f"python3 CSQA.py --num_positive {num_positive} --num_negative {num_negative} --temprature {temprature} --save_path {save_path} --SC_times 5 --data_path pos_neg_30"
    print(cmd)
    os.system(cmd)
    file_path = os.path.join("results/pos_neg_30", save_path)
    lines = load_data(file_path)
    pred_list, gt_list, confi_list = process_data(lines)
    acc, ece, auc = get_results(pred_list, gt_list, confi_list)

    # 将结果放入线程安全的队列中
    result_queue.put({'num_positive': num_positive, 'num_negative': num_negative, 'temprature': temprature, 'counters': counter, 'acc': acc, 'ece': ece, 'auc': auc})

# 创建一个新的CSV文件并写入列名
with open('results.csv', 'w', newline='') as csvfile:
    fieldnames = ['num_positive', 'num_negative', 'temprature', 'counters', 'acc', 'ece', 'auc']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

temprature = 0.7
counters = range(1, 6)
combinations = [(2 * i, 8 - 2 * i) for i in range(5)]

result_queue = queue.Queue()

# 使用线程池执行
with concurrent.futures.ThreadPoolExecutor() as executor:
    for num_positive, num_negative in combinations:
        for counter in counters:
            executor.submit(process_combination, num_positive, num_negative, counter, temprature, result_queue)

# 在所有工作线程完成后，主线程将结果从队列中取出并写入文件
while not result_queue.empty():
    result = result_queue.get()
    with open('results.csv', 'a', newline='') as csvfile:
        fieldnames = ['num_positive', 'num_negative', 'temprature', 'counters', 'acc', 'ece', 'auc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(result)
