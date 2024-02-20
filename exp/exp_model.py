from data_load.dataloader import DataLoader
from explain_module.util import summarize_trial, remove_reflections, save_results#, save_agents
from explain_module.agents import PredictReflectAgent
from predict_module.merge_peft_adapter import merge_peft_adapter
from predict_module.supervised_finetune import supervised_finetune
from predict_module.train_reward_model import train_reward_model
from predict_module.tuning_lm_with_rl import tuning_lm_with_rl
from transformers import LlamaTokenizer, pipeline #, AutoModelForCausalLM, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead
import os, json


class Exp_Model:
    def __init__(self, args):
        self.args = args
        self.dataloader = DataLoader(args)


    def train(self):
        # Collect demonstration data
        print("Loading Train Agents...")
        data = self.dataloader.load(flag="train")

        agent_cls = PredictReflectAgent
        agents = [agent_cls(row['ticker'], row['summary'], row['target']) for _, row in data.iterrows()]
        print("Loaded Train Agents.")

        for agent in agents:
            agent.run()

            if agent.is_correct():
                prompt = agent._build_agent_prompt()
                response = agent.scratchpad.split('Price Movement: ')[-1]

                sample = {"instruction": prompt, "input": "", "output": response}
                with open(self.args.data_path, 'a') as f:
                    f.write(json.dumps(sample) + "\n")

        correct, incorrect = summarize_trial(agents)
        print(f'Finished Trial 0, Correct: {len(correct)}, Incorrect: {len(incorrect)}')

        # Train supervised policy
        supervised_finetune(self.args)
        merge_peft_adapter(model_name=self.args.output_path, output_name=self.args.rl_base_model)

        # Collect comparison data
        comparison_data = []

        for trial in range(self.args.num_reflect_trials):
            for idx, agent in enumerate([a for a in agents if not a.is_correct()]):
                prev_response = agent.scratchpad.split('Price Movement: ')[-1]
                agent.run()

                if agent.is_correct():
                    print(agent._build_agent_prompt(), "\n\n\n")
                    prompt = remove_reflections(agent._build_agent_prompt())
                    response = agent.scratchpad.split('Price Movement: ')[-1]

                    sample = {"user_input": prompt, "completion_a": prev_response, "completion_b": response}
                    comparison_data.append(sample)

            correct, incorrect = summarize_trial(agents)
            print(f'Finished Trial {trial+1}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')

        os.makedirs(self.args.datasets_dir, exist_ok=True)
        comparison_data_path = os.path.join(self.args.datasets_dir, "comparison_data.json")

        if comparison_data:
            with open(comparison_data_path, 'w') as f:
                f.write(json.dumps(comparison_data))

        # Train reward model
        train_reward_model(self.args)
        merge_peft_adapter(model_name=self.args.reward_adapter, output_name=self.args.reward_model_name)

        # Optimize using reinforcement learning
        tuning_lm_with_rl(self.args)
        merge_peft_adapter(model_name=self.args.output_dir+"step_saved", output_name="./saved_models/sep_model")


    def test(self):
        print("Loading Test Agents...")
        data = self.dataloader.load(flag="test")

        agent_cls = PredictReflectAgent
        test_agents = [agent_cls(row['ticker'], row['summary'], row['target']) for _, row in data.iterrows()]
        print("Loaded Test Agents.")

        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            "./saved_models/sep_model",
            load_in_4bit=True,
            device_map="auto"
        )
        tokenizer = LlamaTokenizer.from_pretrained(self.args.output_dir+"step_saved")
        reward_model = pipeline(
            "sentiment-analysis",
            model=self.args.reward_model_name,
            device_map="auto",
            model_kwargs={"load_in_4bit": True},
            tokenizer=tokenizer
        )

        for agent in test_agents:
            agent.run_n_shots(
                              model=model,
                              tokenizer=tokenizer,
                              reward_model=reward_model,
                              num_shots=self.args.num_shots
                              )

        correct, incorrect = summarize_trial(test_agents)
        print(f'Finished evaluation, Correct: {len(correct)}, Incorrect: {len(incorrect)}')

        save_results(test_agents, self.args.save_dir)
