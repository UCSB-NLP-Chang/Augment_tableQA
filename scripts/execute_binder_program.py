"""
Multiprocess executing binder programs.
"""

import json
import argparse
import platform, multiprocessing
import os
import time
import numpy as np
import pyrootutils
import random
pyrootutils.setup_root('.project-root', pythonpath=True)

from nsql.nsql_exec import Executor, NeuralDB
from utils.normalizer import post_process_sql
from utils.utils import load_data_split, majority_vote
from utils.evaluator import Evaluator
from nsql.parser import extract_sql_query

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")


def worker_execute(
        pid,
        args,
        dataset,
        nsql_dict
):
    """
    A worker process for execution.
    """
    result_dict = dict()
    apifile = args.api_config_file
    print(f"Process#{pid} using API file {apifile}")
    n_total_samples, n_correct_samples = 0, 0

    for eid, data_item in enumerate(dataset):
        eid = str(eid)
        if eid not in nsql_dict:
            continue
        print(f"Process#{pid}: eid {eid}, wtq-id {data_item['id']}")
        result_dict[eid] = dict()
        result_dict[eid]['question'] = data_item['question']
        result_dict[eid]['gold_answer'] = data_item['answer_text']
        result_dict[eid]['failed_executions'] = []
        n_total_samples += 1
        table = data_item['table']
        title = table['page_title']
        executor = Executor(args, apifile)
        # Execute
        exec_answer_list = []
        nsql_exec_answer_dict = dict()
        nsqls = nsql_dict[eid]['nsqls']
        if args.use_cot:
            nsqls = [extract_sql_query(i) for i in nsqls]
        if len(nsqls) == 0:
            continue
        result_dict[eid]['all_nsqls'] = nsqls
        for idx, nsql in enumerate(nsqls):
            print(f"Process#{pid}: eid {eid}, original_id {data_item['id']}, executing program#{idx}")
            try:
                if len(nsql) == 0:
                    exec_answer_list.append('<error>')
                    continue
                if nsql in nsql_exec_answer_dict:
                    exec_answer = nsql_exec_answer_dict[nsql]
                else:
                    db = NeuralDB(
                        tables=[{"title": title, "table": table}],
                        dataset_name=args.dataset,
                    )
                    nsql = post_process_sql(
                        sql_str=nsql,
                        df=db.get_table_df(),
                        process_program_with_fuzzy_match_on_db=args.process_program_with_fuzzy_match_on_db,
                        table_title=title
                    )
                    exec_answer = executor.nsql_exec(nsql, db, verbose=args.verbose)
                    if isinstance(exec_answer, str):
                        exec_answer = [exec_answer]
                    nsql_exec_answer_dict[nsql] = exec_answer
                exec_answer_list.append(exec_answer)
            except Exception as e:
                print(f"Process#{pid}: Execution error {e}")
                exec_answer = '<error>'
                exec_answer_list.append(exec_answer)
                result_dict[eid]['failed_executions'].append([idx, nsql, str(e)])
        
        # Majority vote to determine the final prediction answer
        pred_answer, pred_answer_nsqls = majority_vote(
            nsqls=nsqls,
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
        eval_dataset = 'wikitq' if 'squall' in args.dataset or 'wikitq' in args.dataset else args.dataset
        score = Evaluator().evaluate(
            pred_answer,
            gold_answer,
            dataset=eval_dataset,
            question=result_dict[eid]['question']
        )
        result_dict[eid]['score'] = score
        result_dict[eid]['nsqls'] = pred_answer_nsqls
        result_dict[eid]['answer_list'] = exec_answer_list
        n_correct_samples += score
        print(f'Process#{pid}: pred answer: {pred_answer}')
        print(f'Process#{pid}: gold answer: {gold_answer}')
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
    args.input_program_file = os.path.join(args.save_dir, args.input_program_file)

    # Load dataset
    start_time = time.time()
    dataset = load_data_split(args.dataset, args.dataset_split)

    if args.dataset == "wikitq" and args.dataset_split == "test":
        dataset = dataset.select(range(0, 4000, 4))

    # Load programs and process as a unified format
    with open(args.input_program_file, 'r') as f:
        data = json.load(f)
        print(f"========== Loaded programs from {args.input_program_file}. ==========")
    nsql_dict = dict()
    for eid, data_dict in data.items():
        if data[eid]['generations']:
            nsqls = data[eid]['generations']
        else:
            nsqls = ['<dummy program>']
        nsql_dict[eid] = {'nsqls': nsqls}
 
    # Split by processes
    nsql_dict_group = [dict() for _ in range(args.n_processes)]

    for idx, eid in enumerate(nsql_dict.keys()):
        nsql_dict_group[(idx + random.randrange(args.n_processes)) % args.n_processes][eid] = nsql_dict[eid]

    print(f"Executing {sum([len(i) for i in nsql_dict_group])} programs with {args.n_processes} processes.")
    # Execute programs
    result_dict = dict()
    worker_results = []

    if args.debug:
        pid = 0
        import pdb; pdb.set_trace()
        worker_results = worker_execute(
            pid,
            args,
            dataset,
            nsql_dict_group[pid],
        )
        result_dict.update(worker_results)
    else:
        pool = multiprocessing.Pool(processes=args.n_processes)
        for pid in range(args.n_processes):

            worker_results.append(pool.apply_async(worker_execute, args=(
                pid,
                args,
                dataset,
                nsql_dict_group[pid]
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
        json.dump(result_dict, f, indent=2)

    print(f'Done. Elapsed time: {time.time() - start_time}')


if __name__ == '__main__':
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    # File path or name
    parser.add_argument('--dataset', type=str, default='missing_squall',
                        choices=['wikitq', 'missing_squall'])
    parser.add_argument('--dataset_split', type=str, default='validation', choices=['train', 'validation', 'test'])
    parser.add_argument('--api_config_file', type=str, default='key.txt')
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--qa_retrieve_pool_file', type=str, default='templates/qa_retrieve_pool/qa_retrieve_pool.json')
    parser.add_argument('--input_program_file', type=str,
                        default='binder_program_tab_fact_validation.json')
    parser.add_argument('--output_program_execution_file', type=str,
                        default='binder_program_execution_tab_fact_validation.json')

    # Multiprocess options
    parser.add_argument('--n_processes', type=int, default=4)

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
                        choices=['simple', 'prob', 'count', 'answer_biased'])
    parser.add_argument('--answer_biased', type=int, default=None,
                        help='The answer to be biased w. answer_biased_weight in majority vote.')
    parser.add_argument('--answer_biased_weight', type=float, default=None,
                        help='The weight of the answer to be biased in majority vote.')
    parser.add_argument('--process_program_with_fuzzy_match_on_db', action='store_true',
                        help='Whether use fuzzy match with db and program to improve on program.')
    parser.add_argument('--use_cot', action='store_true')
    parser.add_argument('--engine', type=str, default='gpt-3.5-turbo-16k-0613')

    parser.add_argument('--prompt_style', type=str, default='create_table_select_3_full_table',
                        choices=['create_table_select_3_full_table',
                                 'create_table_select_full_table',
                                 'create_table_select_3',])
    parser.add_argument('--seed', type=int, default=42)

    # Debugging options
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()
    print("Args info:")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    main()
