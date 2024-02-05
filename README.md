# Augment_tableQA
This is the implementation for the paper: [Augment before You Try: Knowledge-Enhanced Table Question Answering via Table Expansion](https://arxiv.org/abs/2401.15555).

## Requirements
### Environment
Install conda environment by running
```
conda env create -f environment.yml
conda activate augment
```

## Usage
### 1) OpenAI key
Add your openai API keys in key.txt, one for each line.

### 2) Run scripts
The running scripts are provided in `runscripts/`. To run our method, please use `run_augment_finqa.py`, `run_augment_tatqa.py`, and `run_augment_wikitq.py`. The output will be stored in `results/` and the performance will be printed.
Note: We observe that there might be about 1% random performance variation even if we use greedy decoding. You might try to run the code again if you can't get the number reported in the paper.

## References
If you find our work useful for your research, please consider citing our paper:
```
@misc{liu2024augment,
      title={Augment before You Try: Knowledge-Enhanced Table Question Answering via Table Expansion}, 
      author={Yujian Liu and Jiabao Ji and Tong Yu and Ryan Rossi and Sungchul Kim and Handong Zhao and Ritwik Sinha and Yang Zhang and Shiyu Chang},
      year={2024},
      eprint={2401.15555},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

Our implementation is based on the following repos:
* https://github.com/xlang-ai/Binder
* https://github.com/wenhuchen/Program-of-Thoughts
