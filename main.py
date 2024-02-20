from exp.exp_model import Exp_Model
import argparse
import torch
import numpy as np
import random

fix_seed = 100
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='generating')

# load data
parser.add_argument("--price_dir", type=str, default="data/price/preprocessed/")
parser.add_argument("--tweet_dir", type=str, default="data/tweet/raw/")
parser.add_argument("--seq_len", type=int, default=5)

# supervised finetuning
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--data_path", type=str, default="./data/merge_sample.json")
parser.add_argument("--output_path", type=str, default="./saved_models/lora-Vicuna")
parser.add_argument("--model_path", type=str, default="lmsys/vicuna-7b-v1.5-16k")
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--save_steps", type=int, default=200)
parser.add_argument("--resume_from_supervised_checkpoint", type=str, default=None)
parser.add_argument("--ignore_data_skip", type=str, default="False")

# training reward model
parser.add_argument("--num_reflect_trials", type=int, default=2)
parser.add_argument("--datasets_dir", type=str, default="./datasets/")
parser.add_argument('--local_rank', type=int, default=0, help="Used for multi-gpu")
parser.add_argument('--resume_from_reward_checkpoint', type=bool, default=False, help="If you want to resume training where it left off.")
parser.add_argument('--deepspeed', type=str, default=None, help="Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU.")
parser.add_argument('--per_device_train_batch_size', type=int, default=1)
parser.add_argument('--per_device_eval_batch_size', type=int, default=1)
parser.add_argument('--reward_gradient_accumulation_steps', type=int, default=32)
parser.add_argument('--reward_learning_rate', type=float, default=2e-5)
parser.add_argument('--weight_decay', type=int, default=0.001)
parser.add_argument('--reward_base_model', type=str, default="lmsys/vicuna-7b-v1.5-16k", help="The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc.")
parser.add_argument('--bf16', type=bool, default=False, help="This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU.")
parser.add_argument('--num_train_epochs', type=int, default=1, help="The number of training epochs for the reward model.")
parser.add_argument('--train_subset', type=int, default=100000, help="The size of the subset of the training data to use")
parser.add_argument('--eval_subset', type=int, default=50000, help="The size of the subset of the eval data to use")
parser.add_argument('--gradient_checkpointing', type=bool, default=False, help="Enables gradient checkpointing.")
parser.add_argument('--optim', type=str, default="adamw_hf", help="Enables gradient checkpointing.")
parser.add_argument('--lr_scheduler_type', type=str, default="linear", help="The lr scheduler")
parser.add_argument('--reward_adapter', type=str, default="./saved_models/reward_model_vicuna-7b")

# reinforcement learning
parser.add_argument('--rl_base_model', type=str, default="./saved_models/lora-Vicuna-adapter-merged", help="the model name")
parser.add_argument('--tokenizer_name', type=str, default="lmsys/vicuna-7b-v1.5-16k", help="the tokenizer name")
parser.add_argument('--reward_model_name', type=str, default="./saved_models/reward_model_vicuna-7b-adapter-merged", help="the reward model name")
parser.add_argument('--log_with', type=str, default=None, help="use 'wandb' to log with wandb")
parser.add_argument('--rl_learning_rate', type=float, default=1.4e-5, help="the learning rate")
parser.add_argument('--output_max_length', type=int, default=128, help="maximum length for generation")
parser.add_argument('--mini_batch_size', type=int, default=1, help="the PPO minibatch size")
parser.add_argument('--batch_size', type=int, default=1, help="the batch size")
parser.add_argument('--ppo_epochs', type=int, default=4, help="the number of ppo epochs")
parser.add_argument('--rl_gradient_accumulation_steps', type=int, default=1, help="the number of gradient accumulation steps")
parser.add_argument('--adafactor', type=bool, default=False, help="whether to use the adafactor optimizer")
parser.add_argument('--early_stopping', type=bool, default=True, help="whether to early stop")
parser.add_argument('--target_kl', type=float, default=0.1, help="kl target for early stopping")
parser.add_argument('--reward_baseline', type=float, default=0, help="a baseline value that is subtracted from the reward")
parser.add_argument('--batched_gen', type=bool, default=True, help="whether to use the batched text gen")
parser.add_argument('--save_freq', type=int, default=None, help="n steps to save the model")
parser.add_argument('--output_dir', type=str, default="./saved_models/tuning_llama_rl_checkpoints/", help="directory to save the model")
parser.add_argument('--seed', type=int, default=0, help="the seed")

# evaluation
parser.add_argument("--num_shots", type=int, default=4)
parser.add_argument("--save_dir", type=str, default="results/")

args = parser.parse_args()
print('Args in experiment:')
print(args)

exp_model = Exp_Model(args)
exp_model.train()
exp_model.test()
