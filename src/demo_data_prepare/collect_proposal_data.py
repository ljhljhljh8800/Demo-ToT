
import sys
sys.path.append('./')

from src.io_utils import dump_json, dump_jsonl

dict_input2outputs = {}

path_ = "./data/24/24_train.csv"
list_four_nums = []
header = None
with open(path_, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx == 0:
            header = line.strip()
        else:

            numbers = line.strip().split(",")[1].strip().split(" ")
            numbers = [int(w) for w in numbers]
            assert len(numbers) == 4

            list_four_nums.append(numbers)

# print(list_four_nums)

list_four_nums_demos = []
list_three_nums = []
for nums in list_four_nums:

    demos = []
    for i, digit_i in enumerate(nums):
        for j, digit_j in enumerate(nums):
            if i >= j:
                continue

            for op in ["+", "-", "*", "/"]:
                if op == "+":
                    res_ = digit_i + digit_j
                elif op == "-":
                    if digit_i < digit_j:
                        res_ = digit_j - digit_i
                    else:
                        res_ = digit_i - digit_j
                elif op == "*":
                    res_ = digit_i * digit_j
                else:
                    if digit_i < digit_j:
                        res_ = digit_j / digit_i
                    else:
                        res_ = digit_i / digit_j

                res_ = float(str(res_)[: 4])
                if int(res_) == res_:
                    res_ = int(res_)

                if op in ["/", "-"]:
                    expression = f"{digit_j} {op} {digit_i} = {res_}"
                else:
                    expression = f"{digit_i} {op} {digit_j} = {res_}"

                left = [w for idx, w in enumerate(nums) if idx not in [i, j]]
                left = [res_] + left

                expression = expression + f" (left: {' '.join([str(w) for w in left])})"
                # print(expression)
                demos.append(
                    (nums, op, left, expression)
                )

                list_three_nums.append(left)

    dict_input2outputs[' '.join([str(w) for w in nums])] = demos

    list_four_nums_demos.append(demos)

# print(list_four_nums_demos)
# print(list_three_nums)


list_three_nums_demos = []
list_two_nums = []
for nums in list_three_nums:

    demos = []
    for i, digit_i in enumerate(nums):
        for j, digit_j in enumerate(nums):
            if i >= j:
                continue

            for op in ["+", "-", "*", "/"]:
                if op == "+":
                    res_ = digit_i + digit_j
                elif op == "-":
                    if digit_i < digit_j:
                        res_ = digit_j - digit_i
                    else:
                        res_ = digit_i - digit_j
                elif op == "*":
                    res_ = digit_i * digit_j
                else:
                    if digit_i < digit_j and digit_i != 0:
                        res_ = digit_j / digit_i
                    else:
                        res_ = digit_i / digit_j if digit_j != 0 else 0

                res_ = float(str(res_)[: 4])
                if int(res_) == res_:
                    res_ = int(res_)

                if op in ["/", "-"]:
                    expression = f"{digit_j} {op} {digit_i} = {res_}"
                else:
                    expression = f"{digit_i} {op} {digit_j} = {res_}"

                left = [w for idx, w in enumerate(nums) if idx not in [i, j]]
                left = [res_] + left

                expression = expression + f" (left: {' '.join([str(w) for w in left])})"
                # print(expression)
                demos.append(
                    (nums, op, left, expression)
                )

                list_two_nums.append(left)

    dict_input2outputs[' '.join([str(w) for w in nums])] = demos
    list_three_nums_demos.append(demos)

# print(list_three_nums_demos)
# print(list_two_nums)


list_two_nums_demos = []
list_one_nums = []
for nums in list_two_nums:

    demos = []
    for i, digit_i in enumerate(nums):
        for j, digit_j in enumerate(nums):
            if i >= j:
                continue

            for op in ["+", "-", "*", "/"]:
                if op == "+":
                    res_ = digit_i + digit_j
                elif op == "-":
                    if digit_i < digit_j:
                        res_ = digit_j - digit_i
                    else:
                        res_ = digit_i - digit_j
                elif op == "*":
                    res_ = digit_i * digit_j
                else:
                    if digit_i < digit_j and digit_i != 0:
                        res_ = digit_j / digit_i
                    else:
                        res_ = digit_i / digit_j if digit_j != 0 else 0

                res_ = float(str(res_)[: 4])
                if int(res_) == res_:
                    res_ = int(res_)

                if op in ["/", "-"]:
                    expression = f"{digit_j} {op} {digit_i} = {res_}"
                else:
                    expression = f"{digit_i} {op} {digit_j} = {res_}"

                left = [w for idx, w in enumerate(nums) if idx not in [i, j]]
                left = [res_] + left

                expression = expression + f" (left: {' '.join([str(w) for w in left])})"
                # print(expression)
                demos.append(
                    (nums, op, left, expression)
                )

                list_one_nums.append(left)

    dict_input2outputs[' '.join([str(w) for w in nums])] = demos
    list_two_nums_demos.append(demos)

# print(list_two_nums_demos)
# print(list_one_nums)

print("dict_input2outputs: ", len(dict_input2outputs))
dict_proposal_demos = {}
for input_, outputs in dict_input2outputs.items():

    input_str = f"""Input: {input_}"""
    target_str = f"""Possible next steps:
"""
    for out in outputs:
        exp = out[-1]
        print("exp: ", exp)
        target_str += exp + "\n"

    dict_proposal_demos[input_str] = target_str.strip()

# dump_json(
#     dict_input2output,
#     "src/demo_data_prepare/logs/dict_input2output.json"
# )

list_proposal_demos = []
for k_, v_ in dict_proposal_demos.items():
    list_proposal_demos.append(
        {
            "input": k_,
            "output": v_,
        }
    )
dump_jsonl(
    list_proposal_demos,
    "src/demo_data_prepare/logs/list_proposal_demos.json"
)
