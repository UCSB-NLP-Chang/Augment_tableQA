import os

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")
TOKENIZER_FALSE = "export TOKENIZERS_PARALLELISM=false\n"
DATASET = "missing_squall"
DATASPLIT = 'validation'
G_TYPE = "nsql"
N_SHOTS = 8
N_SAMPLE = 1
TEMPERATURE = 0.0
ENGINE = 'gpt-3.5-turbo-1106'
API_FILE = "key.txt"

os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/annotate_binder_program.py --dataset {DATASET} \
--dataset_split {DATASPLIT} \
--prompt_file templates/prompts/wikitq_nsqlcot_ic.txt \
--system_prompt_file templates/prompts/wikitq_nsqlcot_system.txt \
--output_program_file {G_TYPE}_{DATASET}_{DATASPLIT}_{N_SHOTS}_{N_SAMPLE}_{ENGINE}_{TEMPERATURE}.json \
--n_parallel_prompts 1 \
--n_processes 8 \
--n_shots {N_SHOTS} \
--max_generation_tokens 512 \
--max_api_total_tokens 16001 \
--temperature {TEMPERATURE} \
--sampling_n {N_SAMPLE} \
--engine {ENGINE} \
--api_config_file {API_FILE}""")

# -v 
os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/execute_binder_program.py --dataset {DATASET} \
--dataset_split {DATASPLIT} \
--qa_retrieve_pool_file templates/qa_retrieve_pool/qa_retrieve_pool.json \
--input_program_file {G_TYPE}_{DATASET}_{DATASPLIT}_{N_SHOTS}_{N_SAMPLE}_{ENGINE}_{TEMPERATURE}.json \
--output_program_execution_file {G_TYPE}_{DATASET}_{DATASPLIT}_{N_SHOTS}_{N_SAMPLE}_{ENGINE}_{TEMPERATURE}_exec.json \
--vote_method count \
--engine {ENGINE} \
--n_processes 16 \
--api_config_file {API_FILE} \
--use_cot""")
