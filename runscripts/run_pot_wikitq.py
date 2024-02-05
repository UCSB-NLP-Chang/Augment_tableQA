import os

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")
TOKENIZER_FALSE = "export TOKENIZERS_PARALLELISM=false\n"
DATASET = 'wikitq'
DATASPLIT = "test"
G_TYPE = "pot"
N_SHOTS = 8
N_SAMPLE = 1
TEMPERATURE = 0.0
ENGINE = 'gpt-3.5-turbo-1106'
API_FILE = "key.txt"

os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/execute_pot.py --dataset {DATASET} \
--dataset_split {DATASPLIT} \
--prompt_file templates/prompts/wikitq_pot_ic.txt \
--output_program_file 2_{G_TYPE}_{DATASET}_{DATASPLIT}_{N_SHOTS}_{N_SAMPLE}_{ENGINE}_{TEMPERATURE}.json \
--n_parallel_prompts 1 \
--n_processes 8 \
--n_shots {N_SHOTS} \
--max_generation_tokens 256 \
--max_api_total_tokens 16001 \
--engine {ENGINE} \
--api_config_file {API_FILE}""")
