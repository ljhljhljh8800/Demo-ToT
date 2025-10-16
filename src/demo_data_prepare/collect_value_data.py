

import sys
sys.path.append('./')


from src.io_utils import dump_json, dump_jsonl, load_jsonl, load_json

dict_input2output = {}

paths = [
    "src/demo_data_prepare/logs/list_value_histories_0_200.json",
    "src/demo_data_prepare/logs/list_value_histories_35_200.json",
]
for path_ in paths:
    tmp_data = load_json(
        path_
    )
    tmp_accs = load_json(
        path_.replace("list_value_histories", "list_accs")
    )

    # print(len(tmp_data))
    for samp_data, accs in zip(tmp_data, tmp_accs):
        print(accs)
        if sum(accs) == 0:
            continue

        # print(len(samp_data))
        for d_ in samp_data:
            # print("d_: ", len(d_))

            for s in d_:
                # print("s: ", s)
                if not isinstance(s, dict):
                    continue

                input_ = s["value_prompt"].split("\n\n")[-1].strip()
                if "Answer:" in input_:
                    continue

                value_outputs = s["value_outputs"]
                output_ = value_outputs[0].strip()

                dict_input2output[input_] = output_

list_input2output = []
for k_, v_ in dict_input2output.items():
    list_input2output.append(
        {
            "input": k_,
            "output": v_,
        }
    )

dump_jsonl(
    list_input2output,
    "src/demo_data_prepare/logs/list_value_demos.json"
)
