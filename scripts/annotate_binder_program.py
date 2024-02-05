"""
Multiprocess annotating binder programs.
"""

import time
import json
import argparse
import copy
import os
import random
import re
import pyrootutils
pyrootutils.setup_root(".project-root", pythonpath=True)

from typing import List
import platform
import multiprocessing
from transformers import AutoTokenizer

from generation.generator import Generator
from utils.utils import load_data_split
from nsql.database import NeuralDB
from nsql.parser import extract_added_table, extract_required_variable

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")

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
    apifile = args.api_config_file
    generator = Generator(args, api_key_file=apifile, system_prompt_file=args.system_prompt_file)

    print(f"Worker {pid} using api file {apifile}")
    g_dict = dict()
    built_few_shot_prompts = []
    required_patterns = [r'\b((in|from)\s+(the\s+)?report)\b', r'\b(not\s+(in\s+)?(the\s+)?table)\b']

    for idx, g_eid in enumerate(g_eids):
        g_data_item = dataset[g_eid]
        g_dict[g_eid] = {
            'generations': [],
            'ori_data_item': copy.deepcopy(g_data_item)
        }
        db = NeuralDB(
            tables=[{'title': g_data_item['table']['page_title'], 'table': g_data_item['table']}],
            dataset_name=args.dataset,
        )
        g_data_item['table'] = db.get_table_df()
        g_data_item['title'] = db.get_table_title()
        n_shots = args.n_shots
        few_shot_prompt = generator.build_few_shot_prompt_from_file(
            file_path=args.prompt_file,
            n_shots=n_shots
        )
        generate_prompt = '\nNow answer the following question:' if 'gpt' in args.engine else ''
        max_row = 500 if 'gpt' in args.engine else 100
        remain_tokens = 2000 if 'gpt' in args.engine else 1500
        query_table = generator.build_generate_prompt(
            data_item=g_data_item,
            generate_type=(args.generate_type,),
            datasetname=args.dataset,
            report_ahead=args.dataset=='tatqa' or args.dataset=='finqa',
            info_title=db.table_titles[0],
            max_row=max_row
        )
        max_prompt_tokens = args.max_api_total_tokens - args.max_generation_tokens
        while len(tokenizer.tokenize(query_table)) >= max_prompt_tokens - remain_tokens:
            max_row -= 10
            query_table = generator.build_generate_prompt(
                data_item=g_data_item,
                generate_type=(args.generate_type,),
                datasetname=args.dataset,
                report_ahead=args.dataset=='tatqa' or args.dataset=='finqa',
                info_title=db.table_titles[0],
                max_row=max_row
            )
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
        built_few_shot_prompts.append((g_eid, prompt))

        print(f"Process#{pid}: Prompts ready with {len(built_few_shot_prompts)} parallels. Run openai API.")
        response_dict = generator.generate_one_pass(
            prompts=built_few_shot_prompts,
            verbose=args.verbose,
            include_system_prompt=True
        )
        for eid, g_pairs in response_dict.items():
            g_dict[eid]['generations'] = g_pairs
            if args.dataset in ['finqa', 'tatqa']:
                # check if all required variables are extracted
                add_table = extract_added_table(g_pairs[0])
                if add_table is None:
                    extracted = 0
                else:
                    extracted = add_table.count(': ')
                analyses = extract_required_variable(g_pairs[0]).lower()
                required = 0
                for line in analyses.split('\n'):
                    if any([re.search(pattern, line) for pattern in required_patterns]):
                        required += 1
                if extracted < required:
                    feedback_msg = "There are some variables that need to be extracted from the report but not present in the final JSON output. Please make sure all required variables are extracted in the final JSON output."
                    prompts = [prompt, g_pairs[0], feedback_msg]
                    second_try = generator.generate_one_pass(
                        prompts=[(eid, prompts)],
                        verbose=args.verbose,
                        include_system_prompt=True
                    )
                    for eid, g_pairs in second_try.items():
                        g_dict[eid]['generations'] = g_pairs
        built_few_shot_prompts = []

    # Final generation inference
    if len(built_few_shot_prompts) > 0:
        try:
            response_dict = generator.generate_one_pass(
                prompts=built_few_shot_prompts,
                verbose=args.verbose,
                include_system_prompt=True
            )
            for eid, g_pairs in response_dict.items():
                g_dict[eid]['generations'] = g_pairs
        except Exception as e:
            print(f"==================== Process#{pid}: eid#{g_eid}, wtqid#{g_data_item['id']} generation error: {e} ==========")
            for eid, _ in built_few_shot_prompts:
                g_dict[eid]['generations'] = []

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

    # For Wikitq test split, we load 1k examples to test, since it is expensive to test on full set
    if args.dataset == "wikitq" and args.dataset_split == "test":
        dataset = dataset.select(range(0, 4000, 4))
    else:
        dataset = dataset.select(range(min(args.max_sample_num, len(dataset))))

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

    # Save annotation results
    with open(os.path.join(args.save_dir, args.output_program_file), 'w') as f:
        json.dump(g_dict, f, indent=4)

    num_failed = sum([1 for i in g_dict.values() if len(i['generations']) != args.sampling_n])
    print(f"Elapsed time: {time.time() - start_time}")
    print(f"Total examples: {len(g_dict)}")
    print(f"Failed generations: {num_failed}")


if __name__ == '__main__':
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    # File path or name
    parser.add_argument('--dataset', type=str, default='wikitq',
                        choices=['wikitq', 'missing_squall', 'finqa', 'tatqa'])
    parser.add_argument('--dataset_split', type=str, default='validation', choices=['train', 'validation', 'test'])
    parser.add_argument('--api_config_file', type=str, default='key.txt')
    parser.add_argument('--prompt_file', type=str, default='templates/prompts/wikitq_binder.txt')
    parser.add_argument('--system_prompt_file', type=str, default='templates/prompts/default_system.txt')
    parser.add_argument('--output_program_file', type=str,
                        default='binder_program_execution_tab_fact_validation.json')
    parser.add_argument('--annotated_file', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='results/')

    # Multiprocess options
    parser.add_argument('--n_processes', type=int, default=2)

    # Binder program generation options
    parser.add_argument('--prompt_style', type=str, default='create_table_select_3_full_table',
                        choices=['create_table_select_3_full_table',
                                 'create_table_select_full_table',
                                 'create_table_select_3',
                                 'full_table_content',
                                 'create_table',])
    parser.add_argument('--generate_type', type=str, default='nsql',
                        choices=['nsql', 'augment'])
    parser.add_argument('--n_shots', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)

    # Codex options
    parser.add_argument('--engine', type=str, default="gpt-3.5-turbo-1106")
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
    parser.add_argument('--max_sample_num', type=int, default=9999)

    args = parser.parse_args()
    args.stop_tokens = args.stop_tokens.split('||')
    
    print("Annotating Args info:")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    main()
