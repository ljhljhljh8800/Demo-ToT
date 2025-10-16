import itertools
import time

import numpy as np
from functools import partial

from src.models import gpt

def get_value(task, x, y,
              n_evaluate_sample,
              cache_value=True,
              value_demo_retriever=None,
              value_demo_reranker=None,

              ):
    '''
    x: 4 5 6 10
    y: 6 * 4 = 24 (left: 5 10 24)

    '''
    # print("for value prompting: ", x, y)

    value_prompt = task.value_prompt_wrap(x, y,
                                          value_demo_retriever=value_demo_retriever,
                                          value_demo_reranker=value_demo_reranker,
                                          )
    # print("value_prompt: ", value_prompt)
    # print("-" * 25)

    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)

    if cache_value:
        task.value_cache[value_prompt] = (value, (x, y, value_prompt, value_outputs))

    # print("value: ", value)
    return value, {
        "x": x,
        "y": y,
        "value_prompt": value_prompt,
        "value_outputs": value_outputs,
    }


def get_values(task, x, ys,
               n_evaluate_sample, cache_value=True,
               value_demo_retriever=None,
               value_demo_reranker=None,
               ):
    values = []
    local_value_cache = {}
    histories = []
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates 重复的proposal就去掉
            value, history = local_value_cache[y]
        else:    
            value, history = get_value(task, x, y, n_evaluate_sample,
                                       cache_value=cache_value,
                                       value_demo_retriever=value_demo_retriever,
                                       value_demo_reranker=value_demo_reranker,
                                       )
            local_value_cache[y] = (value, history)
            histories.append(history)
        values.append(value)
    return values, histories

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_proposals(task, x, y, propose_demo_retriever=None, propose_demo_reranker=None):
    propose_prompt = task.propose_prompt_wrap(
        x, y,
        propose_demo_retriever=propose_demo_retriever,
        propose_demo_reranker=propose_demo_reranker,
    )
    print("propose_prompt: ", propose_prompt)

    # 一个prompt，生成多个回复1
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    proposals = [w for w in proposals if "Possible next steps" not in w]

    # print("propose_prompt: ", propose_prompt)
    print("proposals: ", proposals)
    return [y + _ + '\n' for _ in proposals], propose_prompt

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    print("prompt_sample: ", prompt_sample)

    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)

    print("sample prompt: ", prompt)
    return [y + _ for _ in samples]


def solve(args, task, idx, to_print=True,
          propose_demo_retriever=None,
          value_demo_retriever=None,
            propose_demo_reranker=None,
            value_demo_reranker=None,
          ):

    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    print("input x : ", x)

    ys = ['']  # current output candidates
    infos = []
    print()
    list_propose_histories = []   # 如果能够得出正确答案，则是可取的demo
    list_value_histories = []    #
    for step in range(task.steps):
        # generation
        if args.method_generate == 'sample':
            new_ys = [
                get_samples(
                    task, x, y,
                    args.n_generate_sample,
                    prompt_sample=args.prompt_sample,
                    stop=task.stops[step])
                for y in ys
            ]

        elif args.method_generate == 'propose':
            # print("input ys : ", ys)
            tmp_ = [
                get_proposals(task, x, y,
                              propose_demo_retriever=propose_demo_retriever,
                              propose_demo_reranker=propose_demo_reranker)
                for y in ys
            ]
            new_ys = [w[0] for w in tmp_]
            propose_prompts = [w[1] for w in tmp_]
            # print("input new_ys : ", new_ys)

        # time.sleep(1)

        for b_idx, (new_y, y, p_prompt) in enumerate(zip(new_ys, ys, propose_prompts)):
            list_propose_histories.append(
                (step, b_idx, y, new_y, p_prompt)
            )

        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))

        # evaluation
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)

        elif args.method_evaluate == 'value':
            values, value_histories = get_values(
                task, x, new_ys, args.n_evaluate_sample,
                value_demo_retriever=value_demo_retriever,
                value_demo_reranker=value_demo_reranker,
            )
            list_value_histories.append(
                value_histories
            )
        # print("values: ", values)

        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()

        elif args.method_select == 'greedy':
            select_ids = sorted(
                ids,
                key=lambda x: values[x],
                reverse=True
            )[:args.n_select_sample]

        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- step --: {step}\n')
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')

        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
        print(f'-- step --: {step}\n')
        print("select_new_ys: ", select_new_ys)
        # time.sleep(3)

    if to_print: 
        print(ys)
    return ys, {'steps': infos}, list_propose_histories, list_value_histories

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}