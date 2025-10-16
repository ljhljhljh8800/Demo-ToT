import argparse
import json
import os

import sys
sys.path.append("./")

from src.ret_icl.retrievers import FaissRetriever
from src.ret_icl.prompts.crosswords import propose_prompt
from tot.models import gpt
from tot.tasks.crosswords import MiniCrosswordsEnv


def parse_args():
    args = argparse.ArgumentParser()

    # 关于model超参
    args.add_argument('--temperature', type=float, default=0.7)

    # task
    args.add_argument('--task', type=str, required=True, choices=['game24', 'text', 'crosswords'])
    args.add_argument('--file', type=str, required=True, default='24_train.csv')

    args.add_argument('--propose_demo_data_path', type=str,
                      default="src/demo_data_prepare/logs/list_proposal_demos.json")
    args.add_argument('--propose_demo_npy_data_path', type=str,
                      default="./src/demo_data_prepare/logs/proposal_demos.npy")
    args.add_argument('--embed_model_path', type=str, default="./resources/BAAI/bge-base-en-v1___5")

    # 路径
    args.add_argument('--result_dir', type=str, default="experiments/run_1/")

    args = args.parse_args()
    return args


def prompt_wrap(obs, propose_demo_retriever=None):

    retrieved_demos = propose_demo_retriever.search_once(
        "Input: " + obs
    )

    demo_str = ""
    for r_d in retrieved_demos["query2query"]:
        print("r_d: ", r_d)
        input_ = r_d["sample"]["input"]
        output_ = r_d["sample"]["output"].lower()
        demo_str += f"{input_}\n{output_}\n\n"

    propose_prompt_copy = copy.copy(propose_prompt)
    propose_prompt_copy = propose_prompt_copy.replace(
        "<placeholder>", demo_str
    )

    propose_prompt_copy = propose_prompt_copy.format(input=obs)
    return propose_prompt_copy


import re
import copy
from tot.models import gpt

def parse_line(input_str):
    # regular expression pattern to match the input string format
    pattern = r'^([hv][1-5])\. ([a-zA-Z]{5,5}) \((certain|high|medium|low)\).*$'

    # use regex to extract the parts of the input string
    match = re.match(pattern, input_str)

    if match:
        # extract the matched groups
        parts = [match.group(1), match.group(2), match.group(3)]
        return parts
    else:
        return None

confidence_to_value = {'certain': 1, 'high': 0.5, 'medium': 0.2, 'low': 0.1}  # TODO: ad hoc

def parse_response(response):
    # split the response into lines
    lines = response.split('\n')

    # parse each line
    parsed_lines = [parse_line(line) for line in lines]

    # filter out the lines that didn't match the format
    parsed_lines = [(line[0].lower() + '. ' + line[1].lower(), confidence_to_value.get(line[2], 0)) for line in parsed_lines if line is not None]

    return parsed_lines if len(parsed_lines) >= 1 else None


def get_candidates_to_scores(env, propose_demo_retriever=None):
    obs = env.render()
    if obs in env.cache:
        print('cache hit')
        return env.cache[obs]
    print('call gpt')
    responses = gpt(
        prompt_wrap(obs, propose_demo_retriever=propose_demo_retriever),
        model=os.environ["model_name"],
        n=8
    )
    candidates_to_scores = {}
    for response in responses:
        parsed_response = parse_response(response)
        if parsed_response:
            for candidate, score in parsed_response:
                candidates_to_scores[candidate] = candidates_to_scores.get(candidate, 0) + score
        # choose candiate with highest score
    # print(sorted(candidates_to_scores.items(), key=lambda x: x[1], reverse=True))
    env.cache[obs] = candidates_to_scores
    return candidates_to_scores


def propose_score(env, idx, propose_demo_retriever=None):
    obs = env.reset(idx)
    done = False
    infos = []
    while not done:
        responses = gpt(
            prompt_wrap(obs, propose_demo_retriever=propose_demo_retriever),
            model=os.environ["model_name"],
            n=5
        )
        candidates_to_scores = {}
        for response in responses:
            parsed_response = parse_response(response)
            if parsed_response:
                for candidate, score in parsed_response:
                    candidates_to_scores[candidate] = candidates_to_scores.get(candidate, 0) + score
        # choose candiate with highest score
        print(sorted(candidates_to_scores.items(), key=lambda x: x[1], reverse=True))
        if len(candidates_to_scores) == 0:
            break
        candidates =  sorted(candidates_to_scores, key=candidates_to_scores.get, reverse=True)
        for candidate in candidates:
            env_ = copy.deepcopy(env)
            env_.step(candidate)
            if not any(_ == 2 for _ in env_.status):
                break
        print(candidate)
        # candidate = input()
        obs, r, done, info = env.step(candidate)
        print(obs)
        print(env.steps, info)
        print('-------------------\n\n\n')
        infos.append(info)
    return infos


def dfs(env, actions, infos, time_limit, prune, max_per_state, propose_demo_retriever=None):
    # get candidate thoughts
    candidates_to_scores = get_candidates_to_scores(env, propose_demo_retriever=propose_demo_retriever)
    if len(candidates_to_scores) == 0:
        return 0, [], []
    print(sorted(candidates_to_scores.items(), key=lambda x: x[1], reverse=True))

    # back up current state
    board, status, steps = env.board.copy(), env.status.copy(), env.steps

    # try each candidate
    cnt_per_state = 0
    for action in sorted(candidates_to_scores, key=candidates_to_scores.get, reverse=True):
        obs, r, done, info = env.step(action)
        r = info['r_word']

        if len(infos) < time_limit and env.steps < 10 and not any(_ == 2 for _ in env.status):  # not violating any existing constraints
            cnt_per_state += 1
            if cnt_per_state > max_per_state:
                break
            count = env.prompt_status()
            actions.append(action)

            print(len(infos))
            print(actions)
            print(env.render_board())
            print(info)
            print(count)
            if infos:
                best = max(infos, key=lambda x: x['info']['r_word'])
                print('best', best)
            print('--------------')
            print()

            info = {'total_step': len(infos), 'env_step': env.steps, 'actions': actions.copy(), 'info': info, 'count': count}
            infos.append(info)
            if not prune or count['impossible'] < 1:  # only continue if the current status is possible
                dfs(env, actions, infos, time_limit, prune, max_per_state, propose_demo_retriever=propose_demo_retriever)
            actions.pop()
        env.reset(env.idx, board=board.copy(), status=status.copy(), steps=steps)


def main():

    args = parse_args()
    env = MiniCrosswordsEnv(file=args.file)
    os.makedirs(args.result_dir, exist_ok=True)

    # 加载 demo retriever
    propose_demo_retriever = FaissRetriever(
        args.propose_demo_data_path,
        args.propose_demo_npy_data_path,
        args.embed_model_path
    )

    # dfs with pruning
    infoss = []
    n_samples = env.n
    for i in range(0, n_samples):
        env.reset(i)
        infos = []
        actions = []
        dfs(env, actions, infos, 100, prune=True, max_per_state=3, propose_demo_retriever=propose_demo_retriever)
        infoss.append(infos)
        with open(f'{args.result_dir}/infoss_dfs_prune.json', 'w') as fout:
            json.dump(infoss, fout)


if __name__ == '__main__':

    main()