import os
import random

from transformers import AutoTokenizer

from generation.prompt import OpenAIQAPromptBuilder
from generation.generator import Generator
from retrieval.retriever import OpenAIQARetriever
from retrieval.retrieve_pool import OpenAIQARetrievePool, QAItem

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../../")


class OpenAIQAModel(object):
    def __init__(self, args, apifile='openai0'):
        super().__init__()

        random.seed(42)

        retrieve_pool = OpenAIQARetrievePool(
            data_path=os.path.join(ROOT_DIR, args.qa_retrieve_pool_file)
        )
        self.retriever = OpenAIQARetriever(retrieve_pool)
        self.generator = Generator(args=args, 
                                   api_key_file=apifile) # Just to use its call api function

        self.prompting_method = 'new_db'
        self.answer_split_token: str = ';'
        self.db_mapping_token = "\t"
        if 'gpt' in args.engine:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(ROOT_DIR, "utils", "gpt2"))
            self.num_qa_shots = 8
            self.infinite_rows_len = 50  # If the table contain rows larger than this number, it will be handled rows by rows.
            self.max_tokens = 1024
            self.total_max_tokens = 16000
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.engine)
            self.num_qa_shots = 8
            self.infinite_rows_len = 30  # If the table contain rows larger than this number, it will be handled rows by rows.
            self.max_tokens = 660
            self.total_max_tokens = 4000

    def call_openai_api_completion(self, prompt):
        completion = self.generator._call_llm_api(prompt=prompt,
                                                    max_tokens=self.max_tokens,
                                                    temperature=0,
                                                    top_p=1,
                                                    n=1,
                                                    stop=["\n\n"])
        return completion

    def call_openai_for_completion_text(self, prompt, openai_usage_type="completion"):
        if openai_usage_type == "completion":
            completion = self.call_openai_api_completion(prompt)
            if 'code' in self.generator.engine or 'llama' in self.generator.engine:
                return completion.choices[0].text
            else:
                return completion.choices[0].message.content
        else:
            raise ValueError("The model usage type '{}' doesn't exists!".format(openai_usage_type))

    @staticmethod
    def merge_tables(tables, by='row_id'):
        assert len(set([len(_table['rows']) for _table in tables])) == 1, "Tables must have the same rows!"
        merged_header = [by]
        by_idx = tables[0]['header'].index(by)
        merged_rows = [[_row[by_idx]] for _row in tables[0]['rows']]

        for _table in tables:
            header, rows = _table['header'], _table['rows']
            for col_idx, col in enumerate(header):
                if col == by:
                    continue
                if col in merged_header:
                    # When the column is duplicate, and postfix _0, _1 etc.
                    col = "{}_{}".format(col, merged_header.count(col))
                merged_header.append(col)
                for i, row in enumerate(rows):
                    merged_rows[i].append(row[col_idx])
        return {"header": merged_header, "rows": merged_rows}

    def wrap_with_prompt_for_table_qa(self,
                                      question,
                                      sub_table,
                                      table_title=None,
                                      answer_split_token=None,
                                      qa_type="ans",
                                      prompting_method="new_db",
                                      db_mapping_token="😅",
                                      verbose=True):
        q_prompt = "\nGive a database as shown below:\n{}\n\n".format(
            OpenAIQAPromptBuilder.table2codex_prompt(sub_table, table_title)
        )

        if qa_type == "map":
            q_prompt += "Q: Answer question \"{}\" row by row.".format(question)
            assert answer_split_token is not None
            if prompting_method == "basic":
                q_prompt += " The answer should be a list split by '{}' and have {} items in total.".format(
                    answer_split_token, len(sub_table['rows']))

        elif qa_type == "ans":
            q_prompt += "Q: Answer question \"{}\" for the table.".format(question)
        else:
            raise ValueError("The QA type is not supported!")
        
        if qa_type in ['map', 'ans'] and self.num_qa_shots > 0:
            query_item = QAItem(qa_question=question, table=sub_table, title=table_title)
            retrieved_items = self.retriever.retrieve(item=query_item, num_shots=self.num_qa_shots, qa_type=qa_type)
            few_shot_prompt_list = []
            for item in retrieved_items:
                one_shot_prompt = OpenAIQAPromptBuilder.build_one_shot_prompt(
                    item=item,
                    answer_split_token=answer_split_token,
                    verbose=verbose,
                    prompting_method=prompting_method,
                    db_mapping_token=db_mapping_token,
                )
                few_shot_prompt_list.append(one_shot_prompt)
            prompt = '\n'.join(few_shot_prompt_list[:self.num_qa_shots])
            
            cur_qa_shots = self.num_qa_shots
            while len(self.tokenizer(prompt + q_prompt)['input_ids']) + self.max_tokens > self.total_max_tokens:
                cur_qa_shots -= 1
                if cur_qa_shots == 0:
                    raise ValueError("The prompt is too long to fit in the model!")
                prompt = '\n'.join(few_shot_prompt_list[:cur_qa_shots])

        prompt += q_prompt
        prompt += "\n"
        if qa_type == "map":
            if prompting_method == "basic":
                prompt += "A:"
        elif qa_type == "ans":
            prompt += "A:"

        return prompt

    def qa(self, question, sub_tables, qa_type: str, verbose: bool = True, **args):
        # If it is not a problem API can handle, answer it with a QA model.
        merged_table = OpenAIQAModel.merge_tables(sub_tables)
        if verbose:
            print("Make Question {} on {}".format(question, merged_table))
        if qa_type == "map":
            # Map: col(s) -question> one col

            # Make model make a QA towards a sub-table
            # col(s) -> one col, all QA in one time
            def do_map(_table):
                _prompt = self.wrap_with_prompt_for_table_qa(question,
                                                             _table,
                                                             args['table_title'],
                                                             self.answer_split_token,
                                                             qa_type,
                                                             prompting_method=self.prompting_method,
                                                             db_mapping_token=self.db_mapping_token,
                                                             verbose=verbose)
                completion_str = self.call_openai_for_completion_text(_prompt).lower().strip(' []').strip()

                if verbose:
                    print(f'QA map@ input:\n{_prompt}')
                    print(f'QA map@ output:\n{completion_str}')

                if self.prompting_method == "basic":
                    answers = [_answer.strip(" '").lower() for _answer in
                               completion_str.split(self.answer_split_token)]
                elif self.prompting_method == "new_db":
                    answers = [line.split(self.db_mapping_token) for line in completion_str.split("\n")[:-1]]
                    start_idx = -1
                    for i, line in enumerate(answers):
                        if line[0] == merged_table['header'][1]:
                            start_idx = i + 1
                            break
                    if start_idx == -1:
                        start_idx = 2
                    answers = [line[-1] for line in answers[start_idx:]]
                    # answers = [line.split(self.db_mapping_token)[-1] for line in completion_str.split("\n")[2:-1]]
                else:
                    raise ValueError("No such prompting methods: '{}'! ".format(self.prompting_method))
                
                # if len(answers) == 0:
                #     import pdb; pdb.set_trace()

                return answers

            # Handle infinite rows, rows by rows.
            answers = []
            rows_len = len(merged_table['rows'])
            run_times = int(rows_len / self.infinite_rows_len) if rows_len % self.infinite_rows_len == 0 else int(
                rows_len / self.infinite_rows_len) + 1

            for run_idx in range(run_times):
                _table = {
                    "header": merged_table['header'],
                    "rows": merged_table['rows'][run_idx * self.infinite_rows_len:]
                } if run_idx == run_times - 1 else \
                    {
                        "header": merged_table['header'],
                        "rows": merged_table['rows'][run_idx * self.infinite_rows_len:(run_idx + 1) * self.infinite_rows_len]
                    }

                answers.extend(do_map(_table))
            if verbose:
                print("The map@ openai answers are {}".format(answers))
            # Add row_id in addition for finding to corresponding rows.
            return {"header": ['row_id'] + args['new_col_name_s'],
                    "rows": [[row[0], answer] for row, answer in zip(merged_table['rows'], answers)]}
        elif qa_type == "ans":
            # Ans: col(s) -question> answer
            prompt = self.wrap_with_prompt_for_table_qa(question,
                                                        merged_table,
                                                        args['table_title'],
                                                        prompting_method=self.prompting_method,
                                                        verbose=verbose)
            answers = [self.call_openai_for_completion_text(prompt).lower().strip(' []')]

            if verbose:
                print(f'QA ans@ input:\n{prompt}')
                print(f'QA ans@ output:\n{answers}')

            return answers
        else:
            raise ValueError("Please choose from map and ans in the qa usage!!")
