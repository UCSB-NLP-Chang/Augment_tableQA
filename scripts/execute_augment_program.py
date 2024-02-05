"""
Multiprocess executing binder programs.
"""

import json
import argparse
import platform, multiprocessing
import os
import time
import copy
import numpy as np
import random
import re
import pyrootutils
pyrootutils.setup_root('.project-root', pythonpath=True)
from transformers import AutoTokenizer
import pandas as pd

from generation.generator import Generator
from nsql.nsql_exec import Executor, NeuralDB
from nsql.parser import extract_augmentation_command, extract_sql_query, extract_answers, extract_added_table, extract_units
from utils.utils import load_data_split, majority_vote, floatify_ans
from utils.evaluator import Evaluator
from utils.tatqa_metric import TaTQAEmAndF1

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")


def worker_execute(
        pid,
        args,
        dataset,
        nsql_dict,
        tokenizer
):
    """
    A worker process for execution.
    """
    result_dict = dict()
    n_total_samples, n_correct_samples = 0, 0

    apifile = args.api_config_file
    print(f"Process#{pid} using API file {apifile}")
    generator = Generator(args, api_key_file=apifile)
    em_and_f1 = TaTQAEmAndF1()

    for eid, data_item in enumerate(dataset):
        eid = str(eid)
        if eid not in nsql_dict:
            continue
        print(f"Process#{pid}: eid {eid}, wtq-id {data_item['id']}")
        result_dict[eid] = dict()
        result_dict[eid]['question'] = data_item['question']
        result_dict[eid]['gold_answer'] = data_item['answer_text']
        result_dict[eid]['failed_executions'] = []
        result_dict[eid]['nsqls'] = []
        result_dict[eid]['all_nsqls'] = []
        result_dict[eid]['qid'] = data_item['id']
        n_total_samples += 1
        executor = Executor(args, apifile=apifile)
        exec_answer_list = []
        base_sqls = None
        for idx, nsql in enumerate(nsql_dict[eid]['nsqls']):
            print(f"Process#{pid}: eid {eid}, original_id {data_item['id']}, executing program#{idx}")
            try:
                exec_data_item = copy.deepcopy(data_item)
                table = exec_data_item['table']
                title = table['page_title']
                tables = [{"title": title, "table": table}]
                # Step 0: Add additional table
                if args.dataset == 'finqa' or args.dataset == 'tatqa':
                    try:
                        add_table = extract_added_table(nsql)
                        if add_table is not None:
                            try:
                                add_table = json.loads(add_table)
                            except:
                                # Try to fix the json format
                                add_table = re.sub(r'(?<!["\w])\b([a-zA-Z_0-9][\w\s\.\+\-\*]*)\b(?!\w)', r'"\1"', add_table)
                                add_table = add_table.replace('"None"', 'null')
                                add_table = json.loads(add_table)
                                # Eval the expression
                                for k, v in add_table.items():
                                    try:
                                        add_table[k] = [eval(i) for i in v]
                                    except:
                                        pass
                            for k, v in add_table.items():
                                if not isinstance(v, list):
                                    add_table[k] = [v]
                            # make sure all columns have the same length, otherwise pad with None
                            max_len = max([len(v) for v in add_table.values()])
                            for k, v in add_table.items():
                                if len(v) < max_len:
                                    add_table[k] = v + [None] * (max_len - len(v))
                            # if a column is all None, remove it
                            add_table = {k: v for k, v in add_table.items() if not all([i is None for i in v])}
                            add_df = pd.DataFrame(add_table)
                            add_table = {"header": add_df.columns.tolist(), "rows": add_df.values.tolist(),
                                            'page_title': 't2'}
                            tables.append({"title": 't2', "table": add_table})
                    except:
                        pass
                db = NeuralDB(
                    tables=tables,
                    dataset_name=args.dataset
                )
                # Step 1: Augment table
                if not (args.dataset == 'finqa' or args.dataset == 'tatqa'):
                    try:
                        if 'augmentations' in nsql_dict[eid]:
                            aug_commands = nsql_dict[eid]['augmentations'][idx]
                            aug_commands = [[i[2], i[0], i[1]] for i in aug_commands]
                            for aug_i in aug_commands:
                                if isinstance(aug_i[2], str):
                                    aug_i[2] = [aug_i[2]]
                        else:
                            aug_commands = extract_augmentation_command(nsql, db.get_table_df())

                        if len(aug_commands) > 0:
                            executor.augmentation_exec(aug_commands, db, db.table_titles[0], verbose=args.verbose)
                    except Exception as e:
                        pass
                
                # Step 2: Generate SQL code
                n_shots = args.n_shots
                few_shot_prompt = generator.build_few_shot_prompt_from_file(
                    file_path=args.sql_prompt_file,
                    n_shots=n_shots
                )
                if args.dataset in ['finqa', 'tatqa']:
                    generate_prompt = ''
                else:
                    generate_prompt = '\nRead the following table and write a SQL query to answer the question:'
                for ind in range(len(tables) - 1):
                    exec_data_item['table'] = db.get_table_df(ind)
                    exec_data_item['title'] = db.get_table_title(ind)
                    generate_prompt += generator.build_generate_prompt(
                        data_item=exec_data_item,
                        generate_type=(args.generate_type,),
                        table_only=True,
                        datasetname=args.dataset,
                        report_ahead=args.dataset == 'finqa' or args.dataset == 'tatqa'
                    )
                exec_data_item['table'] = db.get_table_df(len(tables) - 1)
                exec_data_item['title'] = db.get_table_title(len(tables) - 1)
                max_row = 200 if 'llama' in args.engine else 500
                remain_tokens = 2000 if 'gpt' in args.engine else 1500
                query_table = generator.build_generate_prompt(
                    data_item=exec_data_item,
                    generate_type=(args.generate_type,),
                    datasetname=args.dataset,
                    report_ahead=(args.dataset == 'finqa' or args.dataset == 'tatqa') and len(tables) == 1,
                    info_title=db.table_titles[0] if len(tables) == 1 else None,
                    max_row=max_row
                )
                max_prompt_tokens = args.max_api_total_tokens - args.max_generation_tokens
                while len(tokenizer.tokenize(query_table)) >= max_prompt_tokens - remain_tokens:
                    max_row -= 10
                    query_table = generator.build_generate_prompt(
                        data_item=exec_data_item,
                        generate_type=(args.generate_type,),
                        datasetname=args.dataset,
                        report_ahead=(args.dataset == 'finqa' or args.dataset == 'tatqa') and len(tables) == 1,
                        info_title=db.table_titles[0] if len(tables) == 1 else None,
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
                        file_path=args.sql_prompt_file,
                        n_shots=n_shots
                    )
                    prompt = few_shot_prompt + "\n\n" + generate_prompt

                    prompt_text = prompt

                print(f"Process#{pid}: Building prompt for eid#{eid}, original_id#{exec_data_item['id']}")
                built_few_shot_prompts = [(eid, prompt)]

                print(f"Process#{pid}: Prompts ready with {len(built_few_shot_prompts)} parallels. Run openai API.")
                need_api_call = True
                
                if len(nsql_dict[eid]['nsqls']) > 1 and len(aug_commands) == 0:
                    if base_sqls is None:
                        args.sampling_n = len(nsql_dict[eid]['nsqls']) * args.sampling_n_per_table
                    else:
                        need_api_call = False
                        nsqls = base_sqls[idx*args.sampling_n_per_table:(idx+1)*args.sampling_n_per_table]
                else:
                    args.sampling_n = args.sampling_n_per_table
                
                if need_api_call:
                    response_dict = generator.generate_one_pass(
                        prompts=built_few_shot_prompts,
                        verbose=args.verbose,
                        include_system_prompt=False
                    )
                    nsqls = [extract_sql_query(i) for i in response_dict[eid]]
                    if args.dataset == 'tatqa':
                        units = [extract_units(i) for i in response_dict[eid]]
                    model_histories = [[i] for i in response_dict[eid]]
                    
                    if len(nsql_dict[eid]['nsqls']) > 1 and len(aug_commands) == 0:
                        assert base_sqls is None
                        base_sqls = nsqls
                        nsqls = nsqls[idx*args.sampling_n_per_table:(idx+1)*args.sampling_n_per_table]
                
                args.sampling_n = args.sampling_n_per_table
                assert len(nsqls) == args.sampling_n_per_table
                result_dict[eid]['nsqls'].extend(nsqls)
                result_dict[eid]['all_nsqls'].extend(nsqls)
                exec_histories = [[] for _ in range(len(nsqls))]
                # Step 3: Execute SQL code
                for sql_ind, nsql in enumerate(nsqls):
                    try:
                        sub_table = db.execute_query(nsql)
                        exec_answer = extract_answers(sub_table)
                        if isinstance(exec_answer, str):
                            exec_answer = [exec_answer]
                        if args.dataset == 'finqa':
                            exec_answer = [floatify_ans(exec_answer)]
                        exec_answer_list.append(exec_answer)
                    except Exception as e:
                        print(f"Process#{pid}\nError when executing SQL query #{sql_ind}: Execution error {e}")
                        exec_histories[sql_ind].append(str(e))
                        exec_answer = '<error>'
                        exec_answer_list.append(exec_answer)
                        result_dict[eid]['failed_executions'].append(
                            [idx*args.sampling_n+sql_ind, nsql, exec_histories[sql_ind][-1]])
            except Exception as e:
                print(f"Process#{pid}\nError when augmenting table\nAugmentation error {e}")
                exec_answer = ['<error>'] * args.sampling_n
                exec_answer_list.extend(exec_answer)
                result_dict[eid]['nsqls'].extend(['<error>'] * args.sampling_n)
                result_dict[eid]['all_nsqls'].extend(['<error>'] * args.sampling_n)
        
        assert len(exec_answer_list) >= len(nsql_dict[eid]['nsqls']) * args.sampling_n_per_table
        # Majority vote to determine the final prediction answer
        pred_answer, pred_answer_nsqls = majority_vote(
            nsqls=result_dict[eid]['nsqls'],
            pred_answer_list=exec_answer_list,
            allow_none_and_empty_answer=args.allow_none_and_empty_answer,
            answer_placeholder=args.answer_placeholder,
            vote_method=args.vote_method,
            answer_biased=args.answer_biased,
            answer_biased_weight=args.answer_biased_weight
        )
        # Evaluate
        result_dict[eid]['pred_answer'] = pred_answer
        gold_answer = data_item['answer_text']
        if args.dataset == 'tatqa' and len(gold_answer) == 2 and gold_answer[-1] == '###':
            gold_answer = gold_answer[0]
        print(f'Process#{pid}: pred answer: {pred_answer}')
        print(f'Process#{pid}: gold answer: {gold_answer}')
        if args.dataset == 'tatqa':
            data_item['answer'] = gold_answer
            if pred_answer == '<error|empty>':
                pred_units = ''
            else:
                pred_units = units[exec_answer_list.index(pred_answer)]
            result_dict[eid]['pred_scale'] = pred_units
            score = em_and_f1(ground_truth=data_item, prediction=pred_answer, pred_scale=pred_units)
        else:
            eval_dataset = 'wikitq' if 'squall' in args.dataset or 'wikitq' in args.dataset else args.dataset
            score = Evaluator().evaluate(
                pred_answer,
                gold_answer,
                dataset=eval_dataset,
                question=result_dict[eid]['question']
            )
        result_dict[eid]['nsqls'] = pred_answer_nsqls
        result_dict[eid]['answer_list'] = exec_answer_list
        result_dict[eid]['score'] = score
        result_dict[eid]['exec_histories'] = exec_histories
        n_correct_samples += score
        if score == 1:
            print(f'Process#{pid}: Correct!')
        else:
            print(f'Process#{pid}: Wrong.')
        print(f'Process#{pid}: Accuracy: {n_correct_samples}/{n_total_samples} = {n_correct_samples / n_total_samples}')

    return result_dict


def main():
    # Build paths
    args.api_config_file = os.path.join(ROOT_DIR, args.api_config_file)
    args.save_dir = os.path.join(ROOT_DIR, args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"========== Using sql prompt file: {args.sql_prompt_file} ==========")

    # Load dataset
    start_time = time.time()
    dataset = load_data_split(args.dataset, args.dataset_split)

    # For Wikitq test split, we load 1k examples to test, since it is expensive to test on full set
    if args.dataset == "wikitq" and args.dataset_split == "test":
        dataset = dataset.select(range(0, 4000, 4))

    # Load programs and process as a unified format
    with open(os.path.join(args.save_dir, args.input_program_file), 'r') as f:
        data = json.load(f)
        print(f"============ Loaded {len(data)} programs from {args.input_program_file}. ============")
    nsql_dict = dict()
    for eid, data_dict in data.items():
        if data[eid]['generations']:
            nsqls = data[eid]['generations']
        else:
            nsqls = ['<dummy program>']
        nsql_dict[eid] = {'nsqls': nsqls}
    
    # Parsed augmentation for Binder-separate
    if args.augmented_file is not None:
        with open(os.path.join(args.save_dir, args.augmented_file), 'r') as f:
            augmented = json.load(f)
        for k, v in augmented.items():
            assert k in nsql_dict
            nsql_dict[k]['augmentations'] = v['augmentations']
    
    nsql_dict_group = [dict() for _ in range(args.n_processes)]
    for idx, eid in enumerate(nsql_dict.keys()):
        nsql_dict_group[(idx + random.randrange(args.n_processes)) % args.n_processes][eid] = nsql_dict[eid]

    print(f"Executing {sum([len(i) for i in nsql_dict_group])} programs with {args.n_processes} processes.")
    # Execute programs
    result_dict = dict()
    worker_results = []

    if 'gpt' in args.engine:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(ROOT_DIR, "utils", "gpt2"))
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.engine)
    if args.debug:
        pid = 0
        import pdb; pdb.set_trace()
        worker_results = worker_execute(
            pid,
            args,
            dataset,
            nsql_dict_group[pid],
            tokenizer, 
        )
        result_dict.update(worker_results)
    else:
        pool = multiprocessing.Pool(processes=args.n_processes)
        for pid in range(args.n_processes):
            worker_results.append(pool.apply_async(worker_execute, args=(
                pid,
                args,
                dataset,
                nsql_dict_group[pid],
                tokenizer
            )))

        # Merge worker results
        for r in worker_results:
            worker_result_dict = r.get()
            result_dict.update(worker_result_dict)
        pool.close()
        pool.join()

    n_correct_samples = 0
    for eid, item in result_dict.items():
        n_correct_samples += item['score']
    print(f'Overall Accuracy: {n_correct_samples}/{len(result_dict)} = {n_correct_samples / len(result_dict)}')

    # Save program executions
    with open(os.path.join(args.save_dir, args.output_program_execution_file), 'w') as f:
        json.dump(result_dict, f)

    print(f'Done. Elapsed time: {time.time() - start_time}')


if __name__ == '__main__':
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    # File path or name
    parser.add_argument('--dataset', type=str, default='wikitq',
                        choices=['wikitq', 'missing_squall', 'finqa', 'tatqa'])
    parser.add_argument('--dataset_split', type=str, default='validation', choices=['train', 'validation', 'test'])
    parser.add_argument('--api_config_file', type=str, default='key.txt')
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--qa_retrieve_pool_file', type=str, default='templates/qa_retrieve_pool/qa_retrieve_pool.json')
    parser.add_argument('--input_program_file', type=str,
                        default='binder_program_tab_fact_validation.json')
    parser.add_argument('--output_program_execution_file', type=str,
                        default='binder_program_execution_tab_fact_validation.json')
    parser.add_argument('--sql_prompt_file', type=str, default='templates/prompts/wikitq_augment_sql.txt')
    parser.add_argument('--system_prompt_file', type=str, default='templates/prompts/default_system.txt')
    parser.add_argument('--augmented_file', type=str, default=None)
    parser.add_argument('--engine', type=str, default='gpt-3.5-turbo-16k-0613')

    # Multiprocess options
    parser.add_argument('--n_processes', type=int, default=2)

    # SQL generation options
    parser.add_argument('--prompt_style', type=str, default='create_table_select_3_full_table',
                        choices=['create_table_select_3_full_table',
                                 'create_table_select_full_table',
                                 'create_table_select_3',
                                 'create_table'])
    parser.add_argument('--generate_type', type=str, default='sql',
                        choices=['nsql', 'augment'])
    parser.add_argument('--n_shots', type=int, default=8)
    parser.add_argument('--max_generation_tokens', type=int, default=512)
    parser.add_argument('--max_api_total_tokens', type=int, default=8001)
    parser.add_argument('--temperature', type=float, default=0.4)
    parser.add_argument('--sampling_n', type=int, default=1)
    parser.add_argument('--sampling_n_per_table', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--stop_tokens', type=str, default='\n\n\n',
                        help='Split stop tokens by ||')
    parser.add_argument('--seed', type=int, default=42)

    # Execution options
    parser.add_argument('--use_majority_vote', action='store_true',
                        help='Whether use majority vote to determine the prediction answer.')
    parser.add_argument('--allow_none_and_empty_answer', action='store_true',
                        help='Whether regarding none and empty executions as a valid answer.')
    parser.add_argument('--allow_error_answer', action='store_true',
                        help='Whether regarding error execution as a valid answer.')
    parser.add_argument('--answer_placeholder', type=str, default='<error|empty>',
                        help='Placeholder answer if execution error occurs.')
    parser.add_argument('--vote_method', type=str, default='count',
                        choices=['simple', 'prob', 'count'])
    parser.add_argument('--answer_biased', type=int, default=None,
                        help='The answer to be biased w. answer_biased_weight in majority vote.')
    parser.add_argument('--answer_biased_weight', type=float, default=None,
                        help='The weight of the answer to be biased in majority vote.')
    parser.add_argument('--process_program_with_fuzzy_match_on_db', action='store_true',
                        help='Whether use fuzzy match with db and program to improve on program.')

    # Debugging options
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    print("Executing Augment Args info:")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    main()
