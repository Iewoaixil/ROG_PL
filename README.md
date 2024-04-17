![image](https://github.com/Iewoaixil/ROG_PL/assets/66995416/73efd318-82dc-4a61-81c5-c1b405ee5592)
# ROG_PL: Robust Open-Set Graph Learning via Region-Based Prototype Learning

## Installation

Before to execute ROG_PL, it is necessary to install the following packages:

* torch
* torch_geometric
* networkx
* matplotlib
* scipy
* numpy
* sklearn
* pyparsing
* faiss-gpu

## Overall Structure

The project is organised as follows:

* `datasets/`contains the necessary dataset files;
* `idx/` contains the noisy dataset index;
* `config/` contains the necessary dataset config;
* `utils/`contains the necessary processing subroutines;
* `Results/`save run results.

## Basic Usage

### Example

```shell
python train.py
```

##Acknowledgement
```shell
This research was supported by National Natural Science Foundation of China (62206179, 92270122), Guangdong Provincial Natural Science Foundation (2022A1515010129, 2023A1515012584), University stability support program of Shenzhen (20220811121315001), Shenzhen Research Foundation for Basic Research, China (JCYJ20210324093000002).
```

##Cite
```shell
@inproceedings{zhang2024rog,
  title={ROG_PL: Robust Open-Set Graph Learning via Region-Based Prototype Learning},
  author={Zhang, Qin and Li, Xiaowei and Lu, Jiexin and Qiu, Liping and Pan, Shirui and Chen, Xiaojun and Chen, Junyang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={4},
  pages={9350--9358},
  year={2024}
}
```



