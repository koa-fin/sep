import openai
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)
# from fastchat.model import get_conversation_template
import torch


class OpenAILLM:
    def __init__(self):
        self.model = "gpt-3.5-turbo-16k"

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def __call__(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        completion = openai.chat.completions.create(model=self.model, messages=messages)
        response = completion.choices[0].message.content
        return response


class FastChatLLM:
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, prompt):
        conv = get_conversation_template('vicuna-7b-1.5')
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        input = conv.get_prompt()

        input_ids = self.tokenizer([input]).input_ids
        output_ids = self.model.generate(
            torch.as_tensor(input_ids).to(self.model.device),
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
        )

        output_ids = output_ids[0][len(input_ids[0]) :]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return response


class NShotLLM:
    def __init__(self, model=None, tokenizer=None, reward_model=None, num_shots=4):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.num_shots = num_shots

    def queries_to_scores(self, list_of_strings):
        return [output["score"] for output in self.reward_model(list_of_strings)]

    def __call__(self, prompt):
        query = self.tokenizer.encode(prompt, return_tensors="pt")
        queries = query.repeat((self.num_shots, 1))
        output_ids = self.model.generate(
            queries,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=1024,
        )
        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        scores = torch.tensor(self.queries_to_scores(output))
        output_ids = output_ids[scores.topk(1).indices[0]][len(query[0]):]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return response
