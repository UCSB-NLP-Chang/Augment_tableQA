from typing import List, Dict

from nsql.qa_module.openai_qa import OpenAIQAModel
from nsql.database import NeuralDB
from nsql.parser import get_cfg_tree, get_steps, remove_duplicate, TreeNode, parse_question_paras, nsql_role_recognize, \
    extract_answers


class Executor(object):
    def __init__(self, args, apifile):
        self.new_col_name_id = 0
        self.qa_model = OpenAIQAModel(args, apifile=apifile)

    def generate_new_col_names(self, number):
        col_names = ["col_{}".format(i) for i in range(self.new_col_name_id, self.new_col_name_id + number)]
        self.new_col_name_id += number
        return col_names

    def sql_exec(self, sql: str, db: NeuralDB, verbose=True):
        if verbose:
            print("Exec SQL '{}' with additional row_id on {}".format(sql, db))
        result = db.execute_query(sql)
        return result

    def nsql_exec(self, nsql: str, db: NeuralDB, verbose=True):
        steps = []
        root_node = get_cfg_tree(nsql)  # Parse execution tree from nsql.
        get_steps(root_node, steps)  # Flatten the execution tree and get the steps.
        steps = remove_duplicate(steps)  # Remove the duplicate steps.
        if verbose:
            print("Steps:", [s.rename for s in steps])
        col_idx = 0
        for step in steps:
            # All steps should be formatted as 'QA()' except for last step which could also be normal SQL.
            assert isinstance(step, TreeNode), "step must be treenode"
            nsql = step.rename
            if nsql.startswith('QA('):
                question, sql_s = parse_question_paras(nsql, self.qa_model)
                sql_executed_sub_tables = []

                # Execute all SQLs and get the results as parameters
                for sql_item in sql_s:
                    role, sql_item = nsql_role_recognize(sql_item,
                                                         db.get_header(),
                                                         db.get_passages_titles(),
                                                         db.get_images_titles())
                    if role in ['col', 'complete_sql']:
                        sql_executed_sub_table = self.sql_exec(sql_item, db, verbose=verbose)
                        sql_executed_sub_tables.append(sql_executed_sub_table)
                    elif role == 'val':
                        val = eval(sql_item)
                        sql_executed_sub_tables.append({
                            "header": ["row_id", "val"],
                            "rows": [["0", val]]
                        })

                if question.lower().startswith("map@"):
                    # When the question is a type of mapping, we return the mapped column.
                    question = question[len("map@"):]
                    if step.father:
                        step.rename_father_col(col_idx=col_idx)
                        sub_table: Dict = self.qa_model.qa(question,
                                                           sql_executed_sub_tables,
                                                           table_title=db.table_title,
                                                           qa_type="map",
                                                           new_col_name_s=step.produced_col_name_s,
                                                           verbose=verbose)
                        db.add_sub_table(sub_table, verbose=verbose)
                        col_idx += 1
                    else:  # This step is the final step
                        sub_table: Dict = self.qa_model.qa(question,
                                                           sql_executed_sub_tables,
                                                           table_title=db.table_title,
                                                           qa_type="map",
                                                           new_col_name_s=["col_{}".format(col_idx)],
                                                           verbose=verbose)
                        return extract_answers(sub_table)

                elif question.lower().startswith("ans@"):
                    # When the question is a type of answering, we return an answer list.
                    question = question[len("ans@"):]
                    answer: List = self.qa_model.qa(question,
                                                    sql_executed_sub_tables,
                                                    table_title=db.table_title,
                                                    qa_type="ans",
                                                    verbose=verbose)
                    if step.father:
                        step.rename_father_val(answer)
                    else:  # This step is the final step
                        return answer
                else:
                    raise ValueError(
                        "Except for operators or NL question must start with 'map@' or 'ans@'!, check '{}'".format(
                            question))

            else:
                sub_table = self.sql_exec(nsql, db, verbose=verbose)
                return extract_answers(sub_table)
    
    def augmentation_exec(self, augment_commands: List[list], db: NeuralDB, table_title: str, verbose=True):
        """Execute augmentation commands on the database."""
        # Execute all augmentation commands.
        for command in augment_commands:
            sql_executed_sub_tables = []

            for sql_item in command[2]:
                sql_executed_sub_table = db.execute_query(sql_item, select_col=True)
                sql_executed_sub_tables.append(sql_executed_sub_table)
        
            # add the new column
            sub_table: Dict = self.qa_model.qa(command[1],
                                                sql_executed_sub_tables,
                                                table_title=table_title,
                                                qa_type="map",
                                                new_col_name_s=[command[0]],
                                                verbose=verbose)
            if len(sub_table['rows']) > 0:
                db.add_sub_table(sub_table, verbose=verbose)
                db.set_db_to_table()
