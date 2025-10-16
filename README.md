

## Model download

```bash
python model_download.py


```



## set up LLM api

```bash

# /root/autodl-tmp/ShapLoRA/resources/LLM-Research/Meta-Llama-3.1-8B-Instruct
nohup vllm serve /root/autodl-tmp/ShapLoRA/resources/LLM-Research/Meta-Llama-3.1-8B-Instruct --gpu_memory_utilization 0.9 --max_model_len 8000 > vllm_log.log & 

nohup vllm serve /root/autodl-tmp/tot_icl/resources/Meta-Llama-3-8B-Instruct --gpu_memory_utilization 0.9 --max_model_len 8000 > vllm_log.log & 

# LLM 起服务
nohup vllm serve ./resources/Qwen2___5-1___5B-Instruct --gpu_memory_utilization 0.75 --max_model_len 8000 > vllm_log.log & 

nohup vllm serve ./resources/Qwen2___5-7B-Instruct --gpu_memory_utilization 0.8 --max_model_len 8000 > vllm_log.log & 

# ./resources/Qwen/Qwen2___5-72B-Instruct-GPTQ-Int4
nohup vllm serve ./resources/Qwen/Qwen2___5-72B-Instruct-GPTQ-Int4 --gpu_memory_utilization 0.95 --max_model_len 5000 > vllm_log.log & 


# /root/autodl-tmp/tot_icl/resources/Qwen/Qwen3-8B
nohup vllm serve /root/autodl-tmp/tot_icl/resources/Qwen/Qwen3-8B --gpu_memory_utilization 0.95 --max_model_len 20000 > vllm_log.log & 


```



## game24 task


collect demo data


```bash

pip install backoff

export OPENAI_API_KEY=123456
export OPENAI_API_BASE="http://localhost:8000/v1"
#export model_name=./resources/tclf90/deepseek-r1-distill-qwen-32b-gptq-int4
export model_name=./resources/Qwen/Qwen2___5-72B-Instruct-GPTQ-Int4
nohup python -u src/demo_data_prepare/run_data_collect.py --task game24 --file 24_train.csv --task_start_index 35 --task_end_index 200 --method_generate propose --method_evaluate value --method_select greedy --n_evaluate_sample 5 --n_select_sample 5 --backend ${model_name} > run_0.log & 



```


small models

(1) fixed demos


```bash

# Meta-Llama-3-8B-Instruct
export TOP_K=8
export OPENAI_API_KEY=123456
export OPENAI_API_BASE="http://localhost:8000/v1"
export model_name=/root/autodl-tmp/tot_icl/resources/Meta-Llama-3-8B-Instruct
nohup python -u tot/run.py --task game24 --file 24_test.csv --task_start_index 0 --task_end_index 25 --method_generate propose --method_evaluate value --method_select greedy --n_evaluate_sample 5 --n_select_sample 5 --backend ${model_name} --result_dir experiments/run_vanilla_1/ > run_1.log & 



# ./resources/Qwen2___5-1___5B-Instruct  start reasoning
export TOP_K=8
export OPENAI_API_KEY=123456
export OPENAI_API_BASE="http://localhost:8000/v1"
export model_name=./resources/Qwen2___5-1___5B-Instruct 
nohup python -u tot/run.py --task game24 --file 24_test.csv --task_start_index 0 --task_end_index 25 --method_generate propose --method_evaluate value --method_select greedy --n_evaluate_sample 5 --n_select_sample 5 --backend ${model_name} --result_dir experiments/run_vanilla_2/ > run_vanilla_2.log & 


# ./resources/Qwen2___5-7B-Instruct 
export TOP_K=8
export OPENAI_API_KEY=123456
export OPENAI_API_BASE="http://localhost:8000/v1"
export model_name=./resources/Qwen2___5-7B-Instruct
nohup python -u tot/run.py --task game24 --file 24_test.csv --task_start_index 0 --task_end_index 25 --method_generate propose --method_evaluate value --method_select greedy --n_evaluate_sample 5 --n_select_sample 5 --backend ${model_name} --result_dir experiments/run_vanilla_3/ > run_vanilla_3.log & 

```




(2) retrieved demos

```bash

# Download embedding model
# python src/model_download.py

cd /root/autodl-fs
cp -r resources/BAAI/ /root/autodl-tmp/tot_icl/resources
cp -r resources/LLM-Research/Meta-Llama-3-8B-Instruct/ /root/autodl-tmp/tot_icl/resources
cp -r resources/Qwen/Qwen2___5-1___5B-Instruct/ /root/autodl-tmp/tot_icl/resources
cp -r resources/Qwen/Qwen2___5-7B-Instruct/ /root/autodl-tmp/tot_icl/resources
cd /root/autodl-tmp/tot_icl


# Collect demo data
python src/demo_data_prepare/collect_proposal_data.py
python src/demo_data_prepare/collect_value_data.py


# Embedding
python src/ret_icl/encode_data.py src/demo_data_prepare/logs/list_proposal_demos.json ./resources/BAAI/bge-base-en-v1___5 src/demo_data_prepare/logs/proposal_demos.npy

python src/ret_icl/encode_data.py src/demo_data_prepare/logs/list_value_demos.json ./resources/BAAI/bge-base-en-v1___5 src/demo_data_prepare/logs/value_demos.npy





# Meta-Llama-3-8B-Instruct to REASONING
export TOP_K=8
export OPENAI_API_KEY=123456
export OPENAI_API_BASE="http://localhost:8000/v1"
export model_name=/root/autodl-tmp/tot_icl/resources/Meta-Llama-3-8B-Instruct
nohup python -u src/ret_icl/run_bfs.py --task game24 --file 24_test.csv --task_start_index 0 --task_end_index 25 --method_generate propose --method_evaluate value --method_select greedy --n_evaluate_sample 5 --n_select_sample 5 --backend ${model_name} --propose_demo_data_path src/demo_data_prepare/logs/list_proposal_demos.json --propose_demo_npy_data_path ./src/demo_data_prepare/logs/proposal_demos.npy --value_demo_data_path src/demo_data_prepare/logs/list_value_demos.json --value_demo_npy_data_path ./src/demo_data_prepare/logs/value_demos.npy --embed_model_path ./resources/bge-base-en-v1___5 --result_dir experiments/run_1/ > run_1.log & 



# ./resources/Qwen2___5-1___5B-Instruct  
export TOP_K=8
export OPENAI_API_KEY=123456
export OPENAI_API_BASE="http://localhost:8000/v1"
export model_name=./resources/Qwen2___5-1___5B-Instruct 
nohup python -u src/ret_icl/run_bfs.py --task game24 --file 24_test.csv --task_start_index 0 --task_end_index 25 --method_generate propose --method_evaluate value --method_select greedy --n_evaluate_sample 5 --n_select_sample 5 --backend ${model_name} --propose_demo_data_path src/demo_data_prepare/logs/list_proposal_demos.json --propose_demo_npy_data_path ./src/demo_data_prepare/logs/proposal_demos.npy --value_demo_data_path src/demo_data_prepare/logs/list_value_demos.json --value_demo_npy_data_path ./src/demo_data_prepare/logs/value_demos.npy --embed_model_path ./resources/bge-base-en-v1___5 --result_dir experiments/run_2/ > run_2.log & 



# ./resources/Qwen2___5-7B-Instruct  
export TOP_K=8
export OPENAI_API_KEY=123456
export OPENAI_API_BASE="http://localhost:8000/v1"
export model_name=./resources/Qwen2___5-7B-Instruct 
nohup python -u src/ret_icl/run_bfs.py --task game24 --file 24_test.csv --task_start_index 0 --task_end_index 25 --method_generate propose --method_evaluate value --method_select greedy --n_evaluate_sample 5 --n_select_sample 5 --backend ${model_name} --propose_demo_data_path src/demo_data_prepare/logs/list_proposal_demos.json --propose_demo_npy_data_path ./src/demo_data_prepare/logs/proposal_demos.npy --value_demo_data_path src/demo_data_prepare/logs/list_value_demos.json --value_demo_npy_data_path ./src/demo_data_prepare/logs/value_demos.npy --embed_model_path ./resources/bge-base-en-v1___5 --result_dir experiments/run_3/ > run_3.log & 





```


(2) with demo reranker

1.5B model

```bash

# Train demo reranker 
# for propose
export TOP_K=32
export model_name=./resources/Qwen2___5-1___5B-Instruct 
nohup python -u src/ret_icl_ft/run_reranker_ft.py --output_dir ./experiments/reranker_run_1/ --demo_data_path ./src/demo_data_prepare/logs/list_proposal_demos.json --embed_model_path ./resources/bge-base-en-v1___5 --bert_model_path ./resources/AI-ModelScope/bert-base-uncased  --learning_rate 10e-4 --gradient_accumulation_steps 2 --warmup_steps 50 --task_type propose --num_epochs 10 > reranker_run_1.log & 

best_test_acc:  0.7633623458666409

use bge embedding model？
python -u src/ret_icl_ft/run_reranker_ft.py --output_dir ./experiments/reranker_run_1/ --demo_data_path ./src/demo_data_prepare/logs/list_proposal_demos.json --embed_model_path ./resources/bge-base-en-v1___5 --bert_model_path ./resources/bge-base-en-v1___5  --learning_rate 10e-4 --gradient_accumulation_steps 2 --warmup_steps 50 --task_type propose --num_epochs 10 

best_test_acc:  0.9886383249004207
 

# for value
export TOP_K=32
export model_name=./resources/Qwen2___5-1___5B-Instruct 
nohup python -u src/ret_icl_ft/run_reranker_ft.py --output_dir ./experiments/reranker_run_2/ --demo_data_path ./src/demo_data_prepare/logs/list_value_demos.json --embed_model_path ./resources/bge-base-en-v1___5 --bert_model_path ./resources/AI-ModelScope/bert-base-uncased  --learning_rate 5e-4 --gradient_accumulation_steps 2 --warmup_steps 50 --task_type value --num_epochs 10 > reranker_run_2.log & 

best_test_acc:  0.7690373471852713

python -u src/ret_icl_ft/run_reranker_ft.py --output_dir ./experiments/reranker_run_2/ --demo_data_path ./src/demo_data_prepare/logs/list_value_demos.json --embed_model_path ./resources/bge-base-en-v1___5 --bert_model_path ./resources/bge-base-en-v1___5  --learning_rate 5e-4 --gradient_accumulation_steps 2 --warmup_steps 50 --task_type value --num_epochs 10

best_test_acc:  1.0


######################
# reranker inference

# Choose demo with the help of DR
# ./resources/Qwen2___5-1___5B-Instruct 
export TOP_K=32
export TOP_K_RERANK=8
export OPENAI_API_KEY=123456
export OPENAI_API_BASE="http://localhost:8000/v1"
export model_name=./resources/Qwen2___5-1___5B-Instruct 
nohup python -u src/ret_icl_ft/run_bfs.py --task game24 --file 24_test.csv --task_start_index 0 --task_end_index 25 --method_generate propose --method_evaluate value --method_select greedy --n_evaluate_sample 5 --n_select_sample 5 --backend ${model_name} --propose_demo_data_path src/demo_data_prepare/logs/list_proposal_demos.json --propose_demo_npy_data_path ./src/demo_data_prepare/logs/proposal_demos.npy --value_demo_data_path src/demo_data_prepare/logs/list_value_demos.json --value_demo_npy_data_path ./src/demo_data_prepare/logs/value_demos.npy --embed_model_path ./resources/bge-base-en-v1___5 --result_dir experiments/reranker_run_tot_1/ --bert_model_path ./resources/bge-base-en-v1___5 --propose_demo_reranker_path ./experiments/reranker_run_1/demo_reranker.bin --value_demo_reranker_path ./experiments/reranker_run_2/demo_reranker.bin > reranker_run_tot_1.log & 




```





7B 模型

```bash

# 训练demo reranker 
# for propose
export TOP_K=32
export model_name=./resources/Qwen2___5-7B-Instruct
# 使用 bge embedding model？
nohup python -u src/ret_icl_ft/run_reranker_ft.py --output_dir ./experiments/reranker_run_7b_1/ --demo_data_path ./src/demo_data_prepare/logs/list_proposal_demos.json --embed_model_path ./resources/bge-base-en-v1___5 --bert_model_path ./resources/bge-base-en-v1___5  --learning_rate 10e-4 --gradient_accumulation_steps 2 --warmup_steps 50 --task_type propose --num_epochs 10 > reranker_run_7b_1.log &

best_test_acc:  1.0

nohup python -u src/ret_icl_ft/run_reranker_ft.py --output_dir ./experiments/reranker_run_7b_1_1/ --demo_data_path ./src/demo_data_prepare/logs/list_proposal_demos.json --embed_model_path ./resources/bge-base-en-v1___5 --bert_model_path ./resources/AI-ModelScope/bert-base-uncased  --learning_rate 10e-4 --gradient_accumulation_steps 2 --warmup_steps 50 --task_type propose --num_epochs 10 > reranker_run_7b_1_1.log &
 
best_test_acc: 


# for value
export TOP_K=32
export model_name=./resources/Qwen2___5-7B-Instruct
nohup python -u src/ret_icl_ft/run_reranker_ft.py --output_dir ./experiments/reranker_run_7b_2/ --demo_data_path ./src/demo_data_prepare/logs/list_value_demos.json --embed_model_path ./resources/bge-base-en-v1___5 --bert_model_path ./resources/bge-base-en-v1___5  --learning_rate 5e-4 --gradient_accumulation_steps 2 --warmup_steps 50 --task_type value --num_epochs 10 > reranker_run_7b_2.log & 

best_test_acc:  0.6863109349082737



##############
# 2025-10-16

# for value
export TOP_K=32
export model_name=./resources/Qwen/Qwen3-8B
nohup python -u src/ret_icl_ft/run_reranker_ft.py --output_dir ./experiments/reranker_run_7b_2_1/ --demo_data_path ./src/demo_data_prepare/logs/list_value_demos.json --embed_model_path ./resources/BAAI/bge-base-en-v1___5 --bert_model_path ./resources/AI-ModelScope/bert-base-uncased  --learning_rate 5e-4 --gradient_accumulation_steps 2 --warmup_steps 50 --task_type value --num_epochs 10 > reranker_run_7b_2_1.log & 

0.xxx

# for propose
export TOP_K=32
export model_name=./resources/Qwen/Qwen3-8B
nohup python -u src/ret_icl_ft/run_reranker_ft.py --output_dir ./experiments/reranker_run_7b_2_2/ --demo_data_path ./src/demo_data_prepare/logs/list_proposal_demos.json --embed_model_path ./resources/BAAI/bge-base-en-v1___5 --bert_model_path ./resources/AI-ModelScope/bert-base-uncased  --learning_rate 5e-4 --gradient_accumulation_steps 2 --warmup_steps 50 --task_type propose --num_epochs 10 > reranker_run_7b_2_2.log & 




######################
# reranker 应用

# 在demo reranker的帮助下，进行demo选择
# ./resources/Qwen2___5-1___5B-Instruct  进行推理
export TOP_K=32
export TOP_K_RERANK=8
export OPENAI_API_KEY=123456
export OPENAI_API_BASE="http://localhost:8000/v1"
export model_name=./resources/Qwen2___5-7B-Instruct 
nohup python -u src/ret_icl_ft/run_bfs.py --task game24 --file 24_test.csv --task_start_index 0 --task_end_index 25 --method_generate propose --method_evaluate value --method_select greedy --n_evaluate_sample 5 --n_select_sample 5 --backend ${model_name} --propose_demo_data_path src/demo_data_prepare/logs/list_proposal_demos.json --propose_demo_npy_data_path ./src/demo_data_prepare/logs/proposal_demos.npy --value_demo_data_path src/demo_data_prepare/logs/list_value_demos.json --value_demo_npy_data_path ./src/demo_data_prepare/logs/value_demos.npy --embed_model_path ./resources/bge-base-en-v1___5 --result_dir experiments/reranker_run_7b_tot_1/ --bert_model_path ./resources/bge-base-en-v1___5 --propose_demo_reranker_path ./experiments/reranker_run_7b_1/demo_reranker.bin --value_demo_reranker_path ./experiments/reranker_run_7b_2/demo_reranker.bin > reranker_run_tot_1.log & 





```
