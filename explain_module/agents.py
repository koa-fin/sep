from typing import List, Union, Literal
from utils.llm import OpenAILLM, NShotLLM #, FastChatLLM
from utils.prompts import REFLECT_INSTRUCTION, PREDICT_INSTRUCTION, PREDICT_REFLECT_INSTRUCTION, REFLECTION_HEADER
from utils.fewshots import PREDICT_EXAMPLES


class PredictAgent:
    def __init__(self,
                 ticker: str,
                 summary: str,
                 target: str,
                 predict_llm = OpenAILLM()
                 ) -> None:

        self.ticker = ticker
        self.summary = summary
        self.target = target
        self.prediction = ''

        self.predict_prompt = PREDICT_INSTRUCTION
        self.predict_examples = PREDICT_EXAMPLES
        self.llm = predict_llm

        self.__reset_agent()

    def run(self, reset=True) -> None:
        if reset:
            self.__reset_agent()

        facts = "Facts:\n" + self.summary + "\n\nPrice Movement: "
        self.scratchpad += facts
        print(facts, end="")

        self.scratchpad += self.prompt_agent()
        response = self.scratchpad.split('Price Movement: ')[-1]
        self.prediction = response.split()[0]
        print(response, end="\n\n\n\n")

        self.finished = True

    def prompt_agent(self) -> str:
        return self.llm(self._build_agent_prompt())

    def _build_agent_prompt(self) -> str:
        return self.predict_prompt.format(
                            ticker = self.ticker,
                            examples = self.predict_examples,
                            summary = self.summary)

    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.target, self.prediction)

    def __reset_agent(self) -> None:
        self.finished = False
        self.scratchpad: str = ''


class PredictReflectAgent(PredictAgent):
    def __init__(self,
                 ticker: str,
                 summary: str,
                 target: str,
                 predict_llm = OpenAILLM(),
                 reflect_llm = OpenAILLM()
                 ) -> None:

        super().__init__(ticker, summary, target, predict_llm)
        self.predict_llm = predict_llm
        self.reflect_llm = reflect_llm
        self.reflect_prompt = REFLECT_INSTRUCTION
        self.agent_prompt = PREDICT_REFLECT_INSTRUCTION
        self.reflections = []
        self.reflections_str: str = ''

    def run(self, reset=True) -> None:
        if self.is_finished() and not self.is_correct():
            self.reflect()

        PredictAgent.run(self, reset=reset)

    def reflect(self) -> None:
        print('Reflecting...\n')
        reflection = self.prompt_reflection()
        self.reflections += [reflection]
        self.reflections_str = format_reflections(self.reflections)
        print(self.reflections_str, end="\n\n\n\n")

    def prompt_reflection(self) -> str:
        return self.reflect_llm(self._build_reflection_prompt())

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
                            ticker = self.ticker,
                            scratchpad = self.scratchpad)

    def _build_agent_prompt(self) -> str:
        prompt = self.agent_prompt.format(
                            ticker = self.ticker,
                            examples = self.predict_examples,
                            reflections = self.reflections_str,
                            summary = self.summary)
        return prompt

    def run_n_shots(self, model, tokenizer, reward_model, num_shots=4, reset=True) -> None:
        self.llm = NShotLLM(model, tokenizer, reward_model, num_shots)
        PredictAgent.run(self, reset=reset)


def format_reflections(reflections: List[str], header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

def EM(prediction, sentiment) -> bool:
    return prediction.lower() == sentiment.lower()
