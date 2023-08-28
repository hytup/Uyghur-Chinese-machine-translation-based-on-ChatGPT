# Uyghur-Chinese-machine-translation-based-on-ChatGPT

## Contents
* [Introduction](#introduction)
* [Usage](#usage)
* [Experimental Results](#experimental-results)
* [License](#License)
* [Citation](#Citation)
* [Development Team](#development-team)
* [Contributors](#contributors)
* [Contact](#contact)

## Introduction

Large Language Model (LLM) technology is now evolving quickly, and ChatGPT, a general-purpose AI model, opens up new opportunities for low-resource machine translation. The GPT-3.5 model, which is optimized using reinforcement learning with human feedback (RLHF), serves as the foundation for ChatGPT. The gpt-3.5-turbo variant, which is reasonably priced and powerful enough for most jobs, is mostly used in this experiment. By conducting a preliminary investigation of Uyghur machine translation tasks using in-context learning (ICL) and chain-of-thought (CoT) methods, this experiment aims to investigate ChatGPT's capabilities on Uyghur-Chinese machine translation tasks in multiple dimensions and lay the groundwork for future academic research.
 

## Usage
Data Sources

Participate in the China Conference on Machine Translation (CCMT) to obtain
 

Train

```
python ICL-zero-shot.py
```



## Experimental Results

ICL zero-shot 
| ID | BLEU | ChrF++ | COMET |
| :------------: | :---: | :--------------: | :----------------: |
| T1      |  13.19 | 11.14 | 78.17 | 
| T2       |  11.17 | 9.93 | 76.50 | 
| T3       |  11.14 | 10.04 | 78.21 | 
| T4      |  10.31 | 9.25 | 75.46 | 


ICL few-shot
| ID | BLEU | ChrF++ | COMET |
| :------------: | :---: | :--------------: | :----------------: |
| 1-shot      |  17.67 | 13.94 | 82.06 | 
| 5-shot       |  17.90 | 14.07 | 82.46 | 
| 10-shot       |  19.82 | 15.33 | 83.46 | 
| 15-shot      |  19.43 | 15.12 | 83.40 | 
| 20-shot      |  18.62 | 14.59 | 82.88 | 

CoT zero-shot
| ID | BLEU | ChrF++ | COMET |
| :------------: | :---: | :--------------: | :----------------: |
| CoT1      |  12.77 | 10.56 | 77.68 | 
| CoT2       |  9.73 | 9.00 | 76.83 | 
| CoT3       |  9.20 | 8.28 | 74.74 | 

CoT few-shot
| ID | BLEU | ChrF++ | COMET |
| :------------: | :---: | :--------------: | :----------------: |
| 1-shot      |  15.87 | 12.72 | 81.75 | 
| 3-shot       |  16.69 | 13.35 | 81.78 | 
| 5-shot       |  17.59 | 13.87 | 82.67 | 
| 10-shot      |  18.92 | 14.75 | 83.32 | 
| 12-shot      |  18.34 | 14.39 | 82.91 | 

## License

The source code is dual licensed. Open source licensing is under the [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause), which allows free use for research purposes. 


## Development Team

Project leaders: Abudukelimu Halidanmu, Abulizi Abudukelimu, Yao Dengfeng

Project members: Hou Yutao, Chen Jishang

## Contributors 
* [Hou Yutao](mailto:hyt1159871021@163.com) 

## Contact

If you have questions, suggestions and bug reports, please email [hyt1159871021@163.com](mailto:hyt1159871021@163.com).
