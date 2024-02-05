"""
Generate nsql and questions.
"""

from typing import Dict, List, Union, Tuple
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI

from generation.prompt import PromptBuilder


@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=1, min=4, max=30))
def call_llm_api(engine, messages, max_tokens, temperature, top_p, n, stop, key):

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


class Generator(object):
    """
    Codex generation wrapper.
    """

    def __init__(self, args, api_key_file: str='key.txt',
                 system_prompt_file: str='templates/prompts/default_system.txt'):
        
        self.args = args
        with open(api_key_file, 'r') as f:
            keys = [line.strip() for line in f.readlines()]
        self.keys = keys
        self.engine = args.engine

        with open(system_prompt_file, 'r') as f:
            self.system_prompt = f.read()

        self.current_key_id = 0

        # if the args provided, will initialize with the prompt builder for full usage
        self.prompt_builder = PromptBuilder(args) if args else None

    def build_few_shot_prompt_from_file(
            self,
            file_path: str,
            n_shots: int
    ):
        """
        Build few-shot prompt for generation from file.
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()
        few_shot_prompt_list = []
        one_shot_prompt = ''
        last_line = None
        for line in lines:
            if line == '\n' and last_line == '\n':
                few_shot_prompt_list.append(one_shot_prompt)
                one_shot_prompt = ''
            else:
                one_shot_prompt += line
            last_line = line
        if len(one_shot_prompt.strip()) > 0:
            few_shot_prompt_list.append(one_shot_prompt)
        few_shot_prompt_list = few_shot_prompt_list[-n_shots:]
        few_shot_prompt_list[-1] = few_shot_prompt_list[
            -1].strip()  # It is essential for prompting to remove extra '\n'
        few_shot_prompt_list = '\n'.join(few_shot_prompt_list)
        return few_shot_prompt_list

    def build_generate_prompt(
            self,
            data_item: Dict,
            generate_type: Tuple,
            table_only: bool = False,
            report_ahead: bool = False,
            datasetname: str = None,
            info_title: str = None,
            max_row: int = None,
    ):
        """
        Build the generate prompt
        """
        return self.prompt_builder.build_generate_prompt(
            **data_item,
            generate_type=generate_type,
            table_only=table_only,
            report_ahead=report_ahead,
            datasetname=datasetname,
            info_title=info_title,
            max_row=max_row
        )

    def generate_one_pass(
            self,
            prompts: List[Tuple],
            verbose: bool = False,
            include_system_prompt: bool = True
    ):
        """
        Generate one pass with codex according to the generation phase.
        """
        response_dict = dict()

        for eid, prompt in prompts:
            result = self._call_llm_api(
                prompt=prompt,
                max_tokens=self.args.max_generation_tokens,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                n=self.args.sampling_n,
                stop=self.args.stop_tokens,
                include_system_prompt=include_system_prompt
            )

            if verbose:
                print('\n', '*' * 20, 'API Call', '*' * 20)
                print(prompt)
                print('\n')
                print('- - - - - - - - - - ->>')

            # parse api results
            for g in result.choices:
                try:
                    if 'code' in self.engine or 'llama' in self.engine:
                        text = g.text
                    else:
                        text = g.message.content
                    eid_pairs = response_dict.get(eid, None)
                    if eid_pairs is None:
                        eid_pairs = []
                        response_dict[eid] = eid_pairs
                    eid_pairs.append(text)

                    if verbose:
                        print(text)

                except Exception as e:
                    print(f'----------- eid {eid} Parsing API results Error --------')
                    print(e)
                    print(g['message'])
                    pass

        return response_dict

    def _call_llm_api(
            self,
            prompt: Union[str, List],
            max_tokens,
            temperature: float,
            top_p: float,
            n: int,
            stop: List[str],
            include_system_prompt: bool = False
    ):
        start_time = time.time()

        key = self.keys[self.current_key_id]
        self.current_key_id = (self.current_key_id + 1) % len(self.keys)
        
        messages = [{"role": "system", "content": self.system_prompt}] if include_system_prompt else []
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        elif isinstance(prompt, list):
            turns = ['user', 'assistant']
            for ind, p in enumerate(prompt):
                messages.append({"role": turns[ind % 2], "content": p})
        
        result = call_llm_api(
            engine=self.engine,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stop=stop,
            key=key,
        )
        
        print('Openai api inference time:', time.time() - start_time)
        return result
