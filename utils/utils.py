"""
General utilities.
"""
import json
import os
from typing import List, Union, Dict
from functools import cmp_to_key
import math
import copy
from collections.abc import Iterable

import recognizers_suite as Recognizers
from recognizers_suite import Culture
from datasets import load_dataset, Dataset, Features, Value, Sequence


ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")

def _load_table(table_path) -> dict:
    """
    attention: the table_path must be the .tsv path.
    Load the WikiTableQuestion from csv file. Result in a dict format like:
    {"header": [header1, header2,...], "rows": [[row11, row12, ...], [row21,...]... [...rownm]]}
    """

    def __extract_content(_line: str):
        _vals = [_.replace("\n", " ").strip() for _ in _line.strip("\n").split("\t")]
        return _vals

    with open(table_path, "r") as f:
        lines = f.readlines()

        rows = []
        for i, line in enumerate(lines):
            line = line.strip('\n')
            if i == 0:
                header = line.split("\t")
            else:
                rows.append(__extract_content(line))

    table_item = {"header": header, "rows": rows}

    # Defense assertion
    for i in range(len(rows) - 1):
        if not len(rows[i]) == len(rows[i - 1]):
            raise ValueError('some rows have diff cols.')

    return table_item


def majority_vote(
        nsqls: List,
        pred_answer_list: List,
        allow_none_and_empty_answer: bool = False,
        allow_error_answer: bool = False,
        answer_placeholder: Union[str, int] = '<error|empty>',
        vote_method: str = 'prob',
        answer_biased: Union[str, int] = None,
        answer_biased_weight: float = None,
):
    """
    Determine the final nsql execution answer by majority vote.
    """

    def _compare_answer_vote_simple(a, b):
        """
        First compare occur times. If equal, then compare max nsql logprob.
        """
        if a[1]['count'] > b[1]['count']:
            return 1
        elif a[1]['count'] < b[1]['count']:
            return -1
        else:
            if a[1]['nsqls'][0][1] > b[1]['nsqls'][0][1]:
                return 1
            elif a[1]['nsqls'][0][1] == b[1]['nsqls'][0][1]:
                return 0
            else:
                return -1

    def _compare_answer_vote_with_prob(a, b):
        """
        Compare prob sum.
        """
        return 1 if sum([math.exp(nsql[1]) for nsql in a[1]['nsqls']]) > sum(
            [math.exp(nsql[1]) for nsql in b[1]['nsqls']]) else -1
    
    def _compare_answer_vote_with_count(a, b):
        """
        Compare occur times.
        """
        if a[1]['count'] > b[1]['count']:
            return 1
        else:
            return -1

    # Vote answers
    candi_answer_dict = dict()
    assert len(nsqls) == len(pred_answer_list)
    for pred_answer, nsql in zip(pred_answer_list, nsqls):
        if allow_none_and_empty_answer:
            if pred_answer == [None] or pred_answer == []:
                pred_answer = [answer_placeholder]
        if allow_error_answer:
            if pred_answer == '<error>':
                pred_answer = [answer_placeholder]

        # Invalid execution results
        if pred_answer == '<error>' or pred_answer == [None] or pred_answer == []:
            continue
        if candi_answer_dict.get(tuple(pred_answer), None) is None:
            candi_answer_dict[tuple(pred_answer)] = {
                'count': 0,
                'nsqls': []
            }
        answer_info = candi_answer_dict.get(tuple(pred_answer), None)
        answer_info['count'] += 1
        answer_info['nsqls'].append(nsql)

    # All candidates execution errors
    if len(candi_answer_dict) == 0:
        return answer_placeholder, []

    # Sort
    if vote_method == 'simple':
        sorted_candi_answer_list = sorted(list(candi_answer_dict.items()),
                                          key=cmp_to_key(_compare_answer_vote_simple), reverse=True)
    elif vote_method == 'count':
        sorted_candi_answer_list = sorted(list(candi_answer_dict.items()),
                                          key=cmp_to_key(_compare_answer_vote_with_count), reverse=True)
    elif vote_method == 'prob':
        sorted_candi_answer_list = sorted(list(candi_answer_dict.items()),
                                          key=cmp_to_key(_compare_answer_vote_with_prob), reverse=True)
    else:
        raise ValueError(f"Vote method {vote_method} is not supported.")

    pred_answer_info = sorted_candi_answer_list[0]
    pred_answer, pred_answer_nsqls = list(pred_answer_info[0]), pred_answer_info[1]['nsqls']
    return pred_answer, pred_answer_nsqls


def load_data_split(dataset_to_load, split, data_dir=os.path.join(ROOT_DIR, 'datasets/')):
    features = {"table": {
        "header": Sequence(Value("string")),
        "rows": Sequence(Sequence(Value("string"))),
        "page_title": Value("string")
        },
        "question": Value("string"),
        "id": Value("string"),
        "answer_text": Sequence(Value("string")),
        "table_id": Value("string"),
        "document_input": Sequence(Value("string"))
    }
    if 'finqa' in dataset_to_load or 'tatqa' in dataset_to_load:
        if 'finqa' in dataset_to_load:
            func = create_finqa_dataset
            features['pre_text'] = Sequence(Value("string"))
            features['post_text'] = Sequence(Value("string"))
        else:
            func = create_tatqa_dataset
            features['answer_type'] = Value("string")
            features['scale'] = Value("string")
        features = Features(features)
        dataset_split_loaded = Dataset.from_generator(func,
            gen_kwargs={'dataset_to_load': dataset_to_load, 'split': split, 'include_all_text': True}, features=features)
        return dataset_split_loaded
    
    dataset_split_loaded = load_dataset(
        path=os.path.join(data_dir, "{}.py".format(dataset_to_load)),
        cache_dir=os.path.join(data_dir, "data"))[split]

    return dataset_split_loaded


def pprint_dict(dic):
    print(json.dumps(dic, indent=2))


def flatten(nested_list):
    for x in nested_list:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def create_finqa_dataset(dataset_to_load, split, include_all_text=False):
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    with open(f'datasets/{dataset_to_load}_{split}.json', 'r') as f:
        data = json.load(f)
    culture = Culture.English
    for each in data:
        d_item = {}
        if '' not in each['table'][0] and len(each['table'][0]) == 2:
            cell = each['table'][0][1].replace('$', '').strip()
            if is_number(cell) and '$' in each['table'][0][1]:
                # add a table header
                each['table'] = [['', 'value']] + each['table']
        d_item['table'] = {'header': each['table'][0], 'rows': each['table'][1:]}
        assert (isinstance(d_item['table']['header'], list) and isinstance(d_item['table']['rows'][0], list))
        assert isinstance(d_item['table']['rows'][0][0], str)
        # Use filename without extension as page_title
        d_item['table']['page_title'] = each['filename'].split('.')[0]
        assert isinstance(d_item['table']['page_title'], str)
        d_item['question'] = each['qa']['question']
        d_item['id'] = each['id']
        d_item['answer_text'] = [each['qa']['answer'].replace('%', ''), str(each['qa']['exe_ans'])]
        # try to recognize currency
        results = Recognizers.recognize_currency(d_item['answer_text'][0], culture)
        if len(results) > 0:
            d_item['answer_text'][0] = results[0].resolution['value']
        d_item['table_id'] = each['filename']
        d_item['pre_text'] = each['pre_text']
        d_item['post_text'] = each['post_text']
        if include_all_text:
            d_item['document_input'] = each['pre_text'] + each['post_text']
        else:
            d_item['document_input'] = [i[1] for i in each['qa']['model_input'] if 'text' in i[0].lower()]
        yield d_item


def create_tatqa_dataset(dataset_to_load, split, include_all_text):
    with open(f'datasets/{dataset_to_load}_{split}.json', 'r') as f:
        data = json.load(f)
    for each in data:
        for ques in each['questions']:
            if ques['answer_from'] != 'table-text':
                continue
            d_item = {}
            remove_first = True
            while remove_first:
                vals = set(each['table']['table'][0])
                if len(vals) == 1 or (len(vals) == 2 and '' in vals):
                    remove_first = True
                else:
                    remove_first = False
                if remove_first:
                    each['table']['table'] = each['table']['table'][1:]
            
            if len(vals) == 3 and '' in vals and len(each['table']['table'][0]) > 4:
                table_ = copy.deepcopy(each['table']['table'])
                for cind, cell in enumerate(table_[0]):
                    if cell != '':
                        end_ind = cind + 1
                        while end_ind < len(table_[0]) and table_[0][end_ind] == '':
                            end_ind += 1
                        for find in range(cind, end_ind):
                            table_[1][find] = cell + ' (' + table_[1][find] + ')'
                d_item['table'] = {'header': table_[1], 'rows': table_[2:]}
            else:
                d_item['table'] = {'header': each['table']['table'][0], 'rows': each['table']['table'][1:]}
            
            assert (isinstance(d_item['table']['header'], list) and isinstance(d_item['table']['rows'][0], list))
            assert isinstance(d_item['table']['rows'][0][0], str)
            d_item['table']['page_title'] = each['table']['uid']
            assert isinstance(d_item['table']['page_title'], str)
            d_item['question'] = ques['question']
            d_item['id'] = ques['uid']
            d_item['answer_text'] = ques['answer']
            if not isinstance(d_item['answer_text'], list):
                d_item['answer_text'] = [d_item['answer_text'], '###']
            d_item['answer_type'] = ques['answer_type']
            d_item['scale'] = ques['scale']
            
            d_item['table_id'] = each['table']['uid']
            d_item['document_input'] = [i['text'] for i in each['paragraphs']]
            yield d_item


def floatify_ans(ans):
    if ans is None:
        return None
    elif type(ans) == bool:
        ans = ans
    elif type(ans) in [list, tuple]:
        if not ans:
            return None
        else:
            try:
                ans = float(ans[0])
            except Exception:
                ans = str(ans[0])
    else:
        try:
            ans = float(ans)
        except Exception:
            ans = str(ans)
    return ans
