"""
Multiprocess annotating binder programs.
"""

import time
import json
import argparse
import copy
import os
import random
import pyrootutils
pyrootutils.setup_root(".project-root", pythonpath=True)

import func_timeout
from typing import List
import platform
import multiprocessing
from transformers import AutoTokenizer
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI

from utils.evaluator import Evaluator
from generation.generator import Generator
from utils.utils import load_data_split
from nsql.database import NeuralDB

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")


def safe_execute(code_string: str, keys=None):
    def execute(x):
        try:
            exec(x)
            locals_ = locals()
            if keys is None:
                return locals_.get('ans', 'Error: No variable named "ans"')
            else:
                return locals_.get(keys, None)
        except Exception as e:
            return "Error: " + str(e)
    try:
        ans = func_timeout.func_timeout(5, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        ans = "Error: Timeout"

    return ans


def parse_api_result(result):
    to_return = []
    for idx, g in enumerate(result.choices):
        text = g.message.content
        text = text.replace('```python\n', '').replace('```Python\n', '').replace('```PYTHON\n', '').replace('```', '').strip()
        to_return.append(text)
    return to_return


@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=1, min=4, max=30))
def call_chatgpt_api(engine, messages, max_tokens, temperature, top_p, n, stop, key):
    client = OpenAI(api_key=key)
    
    result = client.chat.completions.create(
        model=engine,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n,
        stop=stop,
        seed=0,
    )
    
    return result


def linearize_table(table, n_rows=1000):
    lines = [' | '.join(table['table']['header']).strip()]
    for row in table['table']['rows'][:n_rows]:
        lines.append(' | '.join(row).strip())
    lines.append(f'Q: {table["question"]}\n')
    return '\n'.join(lines)


def worker_annotate(
        pid: int,
        args,
        g_eids: List,
        dataset,
        tokenizer
):
    """
    A worker process for annotating.
    """
    with open(args.api_config_file, 'r') as f:
        keys = [line.strip() for line in f.readlines()]
    generator = Generator(args, api_key_file=args.api_config_file)

    g_dict = dict()
    total_num, correct_num = 0, 0

    for idx, g_eid in enumerate(g_eids):
        g_data_item = dataset[g_eid]
        g_dict[g_eid] = {
            'generations': [],
            'ori_data_item': copy.deepcopy(g_data_item)
        }
        n_shots = args.n_shots
        few_shot_prompt = generator.build_few_shot_prompt_from_file(
            file_path=args.prompt_file,
            n_shots=n_shots
        )
        generate_prompt = '\nRead the following table and write a program to answer the question:\n'
        generate_prompt += f'Title: {g_data_item["table"]["page_title"]}\n'
        max_row = 500 if 'gpt' in args.engine else 100
        query_table = linearize_table(g_data_item, n_rows=max_row)
        max_prompt_tokens = args.max_api_total_tokens - args.max_generation_tokens
        while len(tokenizer.tokenize(query_table)) >= max_prompt_tokens - 2000:
            max_row -= 10
            query_table = linearize_table(g_data_item, n_rows=max_row)
        generate_prompt += query_table
        prompt = few_shot_prompt + "\n\n" + generate_prompt

        # Ensure the input length fit Codex max input tokens by shrinking the n_shots
        prompt_text = prompt
        while len(tokenizer.tokenize(prompt_text)) >= max_prompt_tokens:
            n_shots -= 1
            assert n_shots >= 0
            few_shot_prompt = generator.build_few_shot_prompt_from_file(
                file_path=args.prompt_file,
                n_shots=n_shots
            )
            prompt = few_shot_prompt + "\n\n" + generate_prompt
            prompt_text = prompt

        print(f"Process#{pid}: Building prompt for eid#{g_eid}, original_id#{g_data_item['id']}")
        messages = [
            {"role": "user", "content": prompt}
        ]
        result = call_chatgpt_api(
            'gpt-3.5-turbo-1106',
            messages,
            max_tokens=256,
            temperature=0.0,
            top_p=1,
            n=1,
            stop=['\n\n'],
            key=keys[pid % len(keys)]
        )
        codes = parse_api_result(result)
        error_msg = ''
        r = codes[0]
        if 'ans =' in r or 'ans=' in r:
            ans_key = 'ans'
        else:
            ans_key = r.split('\n')[-1].split('=')[0].strip()
        ans = safe_execute(r, keys=ans_key)
        if isinstance(ans, str) and ans.startswith('Error'):
            error_msg = ans
        g_dict[g_eid]['generations'].append(r)
        if isinstance(ans, set):
            ans = list(ans)
        if not isinstance(ans, list):
            ans = [ans]
        
        # Evaluate
        g_dict[g_eid]['pred_answer'] = ans
        gold_answer = g_data_item['answer_text']
        eval_dataset = 'wikitq' if 'squall' in args.dataset or 'wikitq' in args.dataset else args.dataset
        score = Evaluator().evaluate(
            ans,
            gold_answer,
            dataset=eval_dataset,
            question=g_data_item['question']
        )
        g_dict[g_eid]['score'] = score
        g_dict[g_eid]['error_msg'] = error_msg
        if score == 1:
            correct_num += 1
            print(f"Process#{pid}: eid#{g_eid} correct!")
        else:
            print(f"Process#{pid}: eid#{g_eid} wrong!")
        total_num += 1
        print(f"Process#{pid}: {correct_num}/{total_num} = {correct_num / total_num}")

    return g_dict


def main():
    # Build paths
    args.api_config_file = os.path.join(ROOT_DIR, args.api_config_file)
    args.prompt_file = os.path.join(ROOT_DIR, args.prompt_file)
    print(f"==================== Prompt file: {args.prompt_file} ====================")
    args.save_dir = os.path.join(ROOT_DIR, args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    args.stop_tokens = "\n\n" if 'code' in args.prompt_file else "\n\n\n"

    # Load dataset
    start_time = time.time()
    dataset = load_data_split(args.dataset, args.dataset_split)

    if args.dataset == "wikitq" and args.dataset_split == "test":
        dataset = dataset.select(range(0, 4000, 4))

    generate_eids = list(range(len(dataset)))
    generate_eids_group = [[] for _ in range(args.n_processes)]
    for g_eid in generate_eids:
        generate_eids_group[(int(g_eid) + random.randrange(args.n_processes)) % args.n_processes].append(g_eid)
    print('\n******* Annotating *******')
    if 'gpt' in args.engine:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(ROOT_DIR, "utils", "gpt2"))
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.engine)
    
    g_dict = dict()
    worker_results = []
    if args.debug: 
        import pdb; pdb.set_trace()
        res = worker_annotate(
            0,
            args,
            generate_eids_group[0],
            dataset,
            tokenizer
        )
        g_dict.update(res)
    else:
        pool = multiprocessing.Pool(processes=args.n_processes)
        for pid in range(args.n_processes):
            worker_results.append(pool.apply_async(worker_annotate, args=(
                pid,
                args,
                generate_eids_group[pid],
                dataset,
                tokenizer
            )))

        # Merge annotation results
        for r in worker_results:
            worker_g_dict = r.get()
            g_dict.update(worker_g_dict)
        pool.close()
        pool.join()

    num_failed = sum([1 for i in g_dict.values() if len(i['generations']) != args.sampling_n])
    print(f"Elapsed time: {time.time() - start_time}")
    print(f"Total examples: {len(g_dict)}")
    print(f"Failed generations: {num_failed}")
    n_correct_samples = 0
    for eid, item in g_dict.items():
        n_correct_samples += item['score']
    print(f'Overall Accuracy: {n_correct_samples}/{len(g_dict)} = {n_correct_samples / len(g_dict)}')
    
    # Save annotation results
    with open(os.path.join(args.save_dir, args.output_program_file), 'w') as f:
        json.dump(g_dict, f, indent=4)


if __name__ == '__main__':
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    # File path or name
    parser.add_argument('--dataset', type=str, default='wikitq',
                        choices=['wikitq'])
    parser.add_argument('--dataset_split', type=str, default='validation', choices=['train', 'validation', 'test'])
    parser.add_argument('--api_config_file', type=str, default='key.txt')
    parser.add_argument('--prompt_file', type=str, default='templates/prompts/wikitq_binder.txt')
    parser.add_argument('--system_prompt_file', type=str, default='templates/prompts/default_system.txt')
    parser.add_argument('--output_program_file', type=str,
                        default='binder_program_execution_tab_fact_validation.json')
    parser.add_argument('--save_dir', type=str, default='results/')

    # Multiprocess options
    parser.add_argument('--n_processes', type=int, default=2)

    # Binder program generation options
    parser.add_argument('--prompt_style', type=str, default='create_table_select_3_full_table',
                        choices=['create_table_select_3_full_table',
                                 'create_table_select_full_table',
                                 'create_table_select_3',
                                 'full_table_content'])
    parser.add_argument('--generate_type', type=str, default='nsql',
                        choices=['nsql', 'sql'])
    parser.add_argument('--n_shots', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)

    # Codex options
    parser.add_argument('--engine', type=str, default="code-davinci-002")
    parser.add_argument('--n_parallel_prompts', type=int, default=1)
    parser.add_argument('--max_generation_tokens', type=int, default=512)
    parser.add_argument('--max_api_total_tokens', type=int, default=8001)
    parser.add_argument('--temperature', type=float, default=0.4)
    parser.add_argument('--sampling_n', type=int, default=20)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--stop_tokens', type=str, default='\n\n',
                        help='Split stop tokens by ||')

    # debug options
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--max_sample_num', type=int, default=99999999999999)

    args = parser.parse_args()
    args.stop_tokens = args.stop_tokens.split('||')
    
    print("Annotating Args info:")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    main()
