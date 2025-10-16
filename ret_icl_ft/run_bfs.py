import os
import json
import argparse

import sys

import torch

sys.path.append("./")

from src.ret_icl_ft.reranker import RerankerModel
from src.ret_icl_ft.tasks import get_task

from src.ret_icl.retrievers import FaissRetriever
from src.models import gpt_usage

from src.ret_icl_ft.bfs import solve


def run(args):
    task = get_task(args.task, args.file)

    logs, cnt_avg, cnt_any = [], 0, 0

    os.makedirs(args.result_dir, exist_ok=True)
    file = os.path.join(args.result_dir, f"run_logs_{args.task_start_index}_{args.task_end_index}.json")
    file_1 = os.path.join(args.result_dir, f"list_propose_histories_{args.task_start_index}_{args.task_end_index}.json")
    file_2 = os.path.join(args.result_dir, f"list_value_histories_{args.task_start_index}_{args.task_end_index}.json")
    file_3 = os.path.join(args.result_dir, f"list_accs_{args.task_start_index}_{args.task_end_index}.json")

    # 初始化 faiss retriever
    propose_demo_retriever = FaissRetriever(
        args.propose_demo_data_path,
        args.propose_demo_npy_data_path,
        args.embed_model_path
    )
    propose_demo_reranker = RerankerModel(
        bert_model_path=args.bert_model_path
    )
    propose_demo_reranker.load_state_dict(
        torch.load(args.propose_demo_reranker_path),
    )
    value_demo_retriever = FaissRetriever(
        args.value_demo_data_path,
        args.value_demo_npy_data_path,
        args.embed_model_path
    )
    value_demo_reranker = RerankerModel(
        bert_model_path=args.bert_model_path
    )
    value_demo_reranker.load_state_dict(
        torch.load(args.value_demo_reranker_path),
    )

    list_propose_histories = []
    list_value_histories = []
    list_accs = []
    for i in range(args.task_start_index, args.task_end_index):

        # solve
        ys, info, propose_histories, value_histories = solve(
            args, task, i,
            propose_demo_retriever=propose_demo_retriever,
            value_demo_retriever=value_demo_retriever,
            propose_demo_reranker=propose_demo_reranker,
            value_demo_reranker=value_demo_reranker,
        )
        list_propose_histories.append(propose_histories)
        list_value_histories.append(value_histories)

        # log
        infos = [task.test_output(i, y) for y in ys]
        info.update({'idx': i, 'ys': ys, 'infos': infos, 'usage_so_far': gpt_usage(args.backend)})
        logs.append(info)
        with open(file, 'w', encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)

        with open(file_1, 'w', encoding="utf-8") as f_1:
            json.dump(list_propose_histories, f_1, indent=2, ensure_ascii=False)
        with open(file_2, 'w', encoding="utf-8") as f_2:
            json.dump(list_value_histories, f_2, indent=2, ensure_ascii=False)
        
        # log main metric
        accs = [info['r'] for info in infos]
        list_accs.append(
            accs
        )
        with open(file_3, 'w', encoding="utf-8") as f_3:
            json.dump(list_accs, f_3, indent=2, ensure_ascii=False)

        cnt_avg += sum(accs) / len(accs)
        cnt_any += any(accs)
        print("accs: ", accs)
        print(i, 'sum(accs)', sum(accs), 'cnt_avg', cnt_avg / len(logs), 'cnt_any', cnt_any / len(logs), '\n')

        # break
    
    n = args.task_end_index - args.task_start_index
    print(cnt_avg / n, cnt_any / n)
    print('usage_so_far', gpt_usage(args.backend))


def parse_args():
    args = argparse.ArgumentParser()

    # 关于model超参
    args.add_argument('--backend', type=str,
                      # choices=['gpt-4', 'gpt-3.5-turbo'],
                      default='gpt-4')
    args.add_argument('--temperature', type=float, default=0.7)

    # task
    args.add_argument('--task', type=str, required=True, choices=['game24', 'text', 'crosswords'])
    args.add_argument('--file', type=str, required=True, default='24_train.csv')
    args.add_argument('--task_start_index', type=int, default=0)
    args.add_argument('--task_end_index', type=int, default=10000)

    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run

    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote'])
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'], default='greedy')
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)  # 类似于self-consistency
    args.add_argument('--n_select_sample', type=int, default=1)

    # 路径
    args.add_argument('--result_dir', type=str, default="experiments/run_1/")
    args.add_argument('--propose_demo_data_path', type=str, default="src/demo_data_prepare/logs/list_proposal_demos.json")
    args.add_argument('--propose_demo_npy_data_path', type=str, default="./src/demo_data_prepare/logs/proposal_demos.npy")
    args.add_argument('--value_demo_data_path', type=str, default="src/demo_data_prepare/logs/list_value_demos.json")
    args.add_argument('--value_demo_npy_data_path', type=str, default="./src/demo_data_prepare/logs/value_demos.npy")
    args.add_argument('--embed_model_path', type=str, default="./resources/BAAI/bge-base-en-v1___5")

    # reranker
    args.add_argument('--bert_model_path',
                      type=str,
                      default=None)
    args.add_argument('--propose_demo_reranker_path',
                      type=str,
                      default=None)
    args.add_argument('--value_demo_reranker_path',
                      type=str,
                      default=None)

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)