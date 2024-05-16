# Summarize-Explain-Predict (SEP)

This repository contains the code for "Learning to Generate Explainable Stock Predictions using Self-Reflective Large Language Models" [[Paper](https://arxiv.org/abs/2402.03659)].

## Setup

To get started:

1. Install the module dependencies into your environment:
```bash
pip install -r requirements.txt
```

2. Set `OPENAI_API_KEY` environment variable to your OpenAI API key:
```bash
export OPENAI_API_KEY=<your key>
```

3. Run a sample experiment:
```bash
python main.py --price_dir "data/sample_price/preprocessed/" --tweet_dir "data/sample_tweet/raw/"
```

## Note

The full dataset used in the work can be found [here](https://github.com/koa-fin/sn2).

Due to the nature of these experiments, it may not be feasible for individual developers to rerun the full results as OpenAI has significant API charges.

## Citation

If you find this repository useful, please cite our paper.

```
@inproceedings{koa2024learning,
  title={Learning to Generate Explainable Stock Predictions using Self-Reflective Large Language Models},
  author={Koa, Kelvin J.L. and Ma, Yunshan and Ng, Ritchie and Chua, Tat-Seng},
  booktitle={Proceedings of the ACM on Web Conference 2024},
  pages={4304–4315},
  year={2024}
}
```

## Acknowledgement

We appreciate the following GitHub repositories a lot for their valuable code base:

https://github.com/noahshinn/reflexion

https://github.com/jackaduma/Vicuna-LoRA-RLHF-PyTorch
