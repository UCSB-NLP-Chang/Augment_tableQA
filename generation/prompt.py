"""
Build NSQL generation prompt.
Two main parts:
1) PromptBuilder makes prompt for calling codex to generate NSQL(Binder-SQL).
2) OpenAIQAPromptBuilder makes prompt for calling codex to generate QA answers.
"""

import random
from typing import Dict, Tuple, List
import pandas as pd
import copy

from utils.errors import DuplicateColumnsError
from retrieval.retrieve_pool import QAItem

from utils.normalizer import prepare_df_for_neuraldb_from_table


def _create_table_prompt(df: pd.DataFrame, title: str):
    """
    Return the CREATE TABLE clause as prompt.
    """
    string = "CREATE TABLE {}(\n".format(title)
    for header in df.columns:
        column_type = 'text'
        try:
            if df[header].dtype == 'int64':
                column_type = 'int'
            elif df[header].dtype == 'float64':
                column_type = 'real'
            elif df[header].dtype == 'datetime64':
                column_type = 'datetime'
        except AttributeError as e:
            raise DuplicateColumnsError(e)

        string += '\t{} {},\n'.format(header, column_type)
    string = string.rstrip(',\n') + ')\n'
    return string


class PromptBuilder(object):
    def __init__(self, args):
        self.args = args
        self.prompt_style = args.prompt_style
        random.seed(args.seed)

    def _select_x_prompt(self, df: pd.DataFrame, num_rows: int,
                         few_shot_demonstration=True, column_split='\t', table_name='w'):
        """
        Return the first X rows table contents as prompt.
        """
        if self.prompt_style == 'create_table_select_full_table':
            string = '/*\nAll rows of the table:\nSELECT * FROM {};\n'.format(table_name)
        elif self.prompt_style == 'create_table_select_3':
            string = '/*\n{} example rows:\nSELECT * FROM w LIMIT {};\n'.format(num_rows, num_rows)
        elif self.prompt_style == 'full_table_content':
            string = ''
        elif self.prompt_style == 'create_table_select_3_hidden':
            string = '/*\n{} example rows:\n'.format(num_rows)
        elif few_shot_demonstration is True and self.prompt_style in \
                ["create_table_select_3_full_table",
                 "create_table_select_3_full_table_w_gold_passage_image",
                 "create_table_select_3_full_table_w_all_passage_image"]:
            string = '/*\n{} example rows:\nSELECT * FROM w LIMIT {};\n'.format(num_rows, num_rows)
        elif few_shot_demonstration is False and self.prompt_style in \
                ["create_table_select_3_full_table",
                 "create_table_select_3_full_table_w_gold_passage_image",
                 "create_table_select_3_full_table_w_all_passage_image"]:
            string = '/*\nAll rows of the table:\nSELECT * FROM {};\n'.format(table_name)
        else:
            raise ValueError(f"Select x prompt style {self.prompt_style} is not supported.")

        for column_id, header in enumerate(df.columns):
            string += str(header)
            if column_id != len(df.columns) - 1:
                string += column_split
        string += '\n'

        for row in df.iloc[:num_rows].itertuples():
            for column_id in range(1, len(df.columns) + 1):
                string += str(row[column_id])
                if column_id != len(df.columns):
                    string += column_split
            string += '\n'
        
        if self.prompt_style != 'full_table_content':
            string += '*/\n'

        return string

    def build_generate_prompt(
            self,
            generate_type: Tuple,
            table: pd.DataFrame,
            question: str = None,
            title: str = None,
            document_input: List[str] = None,
            table_only: bool = False,
            report_ahead: bool = False,
            datasetname: str = None,
            info_title: str = None,
            max_row: int = None,
            **kwargs
    ):
        """
        Build the prompt of the generation sample.
        """
        generate_prompt = "\n"

        if report_ahead:
            generate_prompt += "Report:\n"
            if len(document_input) > 0:
                for line_i in document_input:
                    generate_prompt += line_i.strip() + ' '
                generate_prompt = generate_prompt[:-1] + '\n'
            else:
                generate_prompt += "Empty\n"
            generate_prompt += "Tables:\n"
        
        # table prompt
        if datasetname in ['wikitq', 'missing_squall']:
            generate_prompt += f'Title: {info_title}\n'
        if self.prompt_style in ['create_table_select_full_table', 'create_table_select_3_full_table']:
            generate_prompt += _create_table_prompt(table, title)
            num_rows = table.shape[0] if max_row is None else min(table.shape[0], max_row)
            generate_prompt += self._select_x_prompt(
                df=table,
                num_rows=num_rows,
                few_shot_demonstration=False,
                table_name=title
            )
        elif self.prompt_style in ['create_table_select_3']:
            generate_prompt += _create_table_prompt(table, title)
            generate_prompt += self._select_x_prompt(
                df=table,
                num_rows=3,
                few_shot_demonstration=False
            )
        elif self.prompt_style in ['full_table_content']:
            generate_prompt += self._select_x_prompt(
                df=table,
                num_rows=table.shape[0],
                few_shot_demonstration=False,
                column_split=' | '
            )
        else:
            raise ValueError('{} is not supported.'.format(self.prompt_style))

        if table_only:
            return generate_prompt
        
        generate_prompt += '\n'
        if self.prompt_style == 'full_table_content':
            generate_prompt += 'Question: {}\n'.format(question)
        else:
            generate_prompt += 'Q: {}\n'.format(question)
        
        # determine the target to generate
        if generate_type == ('answer',):
            generate_prompt += 'A: '
        elif generate_type == ('augment',):
            if self.prompt_style != 'full_table_content':
                generate_prompt += 'Transformation: '
            else:
                generate_prompt += 'Analysis:\n'
        elif generate_type == ('nsql',):
            generate_prompt += 'NeuralSQL: '
        elif generate_type == ('sql',):
            generate_prompt += 'SQL: '
        elif generate_type == ('npython',):
            generate_prompt += 'NeuralPython: '
        elif generate_type == ('python',):
            generate_prompt += 'Python: '
        else:
            raise ValueError(f'Generate type {generate_type} is not supported.')

        return generate_prompt


class OpenAIQAPromptBuilder(object):
    @staticmethod
    def table2codex_prompt(table, table_title=None, drop_row_id=True, ):
        _table = copy.deepcopy(table)
        header = _table['header']
        rows = _table['rows']
        if drop_row_id:
            if header[0] == "row_id":
                header = header[1:]
                rows = [_row[1:] for _row in rows]
        prompt_str = 'Table: {}\n'.format(table_title) if table_title else ''
        prompt_str += "/*\n"
        prompt_str += "\t".join(header) + "\n"
        prompt_str += '\n'.join(["\t".join([str(cell) for cell in row]) for row in rows]) + "\n"
        prompt_str += "*/"
        return prompt_str

    @staticmethod
    def build_one_shot_prompt(
            item: QAItem,
            answer_split_token: str = ';',
            verbose: bool = False,
            prompting_method='new_db',
            db_mapping_token="😅",
    ) -> str:
        """
        Build one-shot QA prompt.
        """
        assert prompting_method in ['basic', 'new_db']
        qa_type, qa_question = item.qa_question.split('@')
        db_prompt = OpenAIQAPromptBuilder.table2codex_prompt(item.table, item.title)
        q_prompt = "Give a database as shown below:\n{}\n\n".format(db_prompt)

        if prompting_method == 'basic':
            if qa_type == "map":
                prompt += "Q: Answer question \"{}\" row by row.".format(qa_question)
                assert answer_split_token is not None
                prompt += " The answer should be a list split by '{}' and have {} items in total.".format(
                    answer_split_token, len(item.table['rows']))
                prompt += "\nA: {}\n\n".format(f'{answer_split_token}'.join(item.qa_answer))
            elif qa_type == "ans":
                prompt += "Q: Answer question \"{}\" for the table.".format(qa_question)
                prompt += " "
                prompt += "\nA: {}\n\n".format(f'{answer_split_token}'.join(item.qa_answer))
            else:
                raise ValueError("The QA type is not supported!")

            return prompt

        elif prompting_method == "new_db":
            if qa_type == "map":
                q_prompt += "Q: Answer question \"{}\" row by row.".format(qa_question)
                assert answer_split_token is not None
                db_prompt_lines = db_prompt.split("\n")[2:-1]  # skip Title, /*, and */
                db_prompt_lines_with_answer = []
                db_prompt_lines_with_answer.append("/*")
                db_prompt_lines_with_answer.append(db_prompt_lines[0])
                assert len(db_prompt_lines[1:]) == len(
                    item.qa_answer), "answer items and table rows must be in the same number, check annotations"
                for db_prompt_line, qa_answer_item in zip(db_prompt_lines[1:], item.qa_answer):
                    db_prompt_lines_with_answer.append(
                        "{}{}{}".format(db_prompt_line, db_mapping_token, qa_answer_item))
                db_prompt_lines_with_answer.append("*/")
                a_prompt = "\n".join(db_prompt_lines_with_answer)

            elif qa_type == "ans":
                q_prompt += "Q: Answer question \"{}\" for the table.".format(qa_question)
                # prompt += " "
                a_prompt = "A: {}".format(f'{answer_split_token}'.join(item.qa_answer))
            else:
                raise ValueError("The QA type is not supported!")
            
            messages = f"{q_prompt}\n{a_prompt}\n"

            return messages
