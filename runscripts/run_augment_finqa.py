import os

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")
TOKENIZER_FALSE = "export TOKENIZERS_PARALLELISM=false\n"
DATASET = 'finqa'
DATASPLIT = "test"
G_TYPE = "augment"
N_SHOTS = 4
N_SAMPLE = 1
AUG_TEMPERATURE = 0.0
SQL_TEMPERATURE = 0.0
ENGINE = 'gpt-3.5-turbo-1106'
API_FILE = "key.txt"

os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/annotate_binder_program.py --dataset {DATASET} \
--dataset_split {DATASPLIT} \
--prompt_file templates/prompts/finqa_augment_ic.txt \
--system_prompt_file templates/prompts/finqa_augment_system.txt \
--prompt_style full_table_content \
--output_program_file {G_TYPE}_{DATASET}_{DATASPLIT}_{N_SHOTS}_{N_SAMPLE}_{ENGINE}_{AUG_TEMPERATURE}.json \
--n_parallel_prompts 1 \
--n_processes 4 \
--generate_type {G_TYPE} \
--n_shots {N_SHOTS} \
--max_generation_tokens 512 \
--max_api_total_tokens 16001 \
--temperature {AUG_TEMPERATURE} \
--sampling_n {N_SAMPLE} \
--engine {ENGINE} \
--api_config_file {API_FILE}""")  # --debug

os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/execute_augment_program.py --dataset {DATASET} \
--dataset_split {DATASPLIT} \
--sql_prompt_file templates/prompts/finqa_sql_ic.txt \
--input_program_file {G_TYPE}_{DATASET}_{DATASPLIT}_{N_SHOTS}_{N_SAMPLE}_{ENGINE}_{AUG_TEMPERATURE}.json \
--output_program_execution_file {G_TYPE}_{DATASET}_{DATASPLIT}_{N_SHOTS}_{N_SAMPLE}_{ENGINE}_{SQL_TEMPERATURE}_exec.json \
--vote_method count \
--engine {ENGINE} \
--n_processes 4 \
--temperature {SQL_TEMPERATURE} \
--n_shots {N_SHOTS} \
--max_generation_tokens 512 \
--max_api_total_tokens 16001 \
--sampling_n_per_table 1 \
--api_config_file {API_FILE}""")
