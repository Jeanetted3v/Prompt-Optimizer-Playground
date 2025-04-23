"""Reference:
https://github.com/zou-group/textgrad/issues/116#issue-2494907041
"""
import json
import os
from typing import List, Union
import textgrad as tg
from textgrad import Variable
from textgrad.engine.openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = ""


class ChatOpenAIWithHistory(ChatOpenAI):
    def __init__(self, *args, **kwargs):
        self.history_messsages = []
        super().__init__(*args, **kwargs)

    def inject_history(self, messages: list[dict]) -> None:
        self.history_messsages = messages

    def _generate_from_single_prompt(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature=0,
        max_tokens=2000,
        top_p=0.99,
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        messages = [
            {"role": "system", "content": sys_prompt_arg},
            *self.history_messsages,
            {"role": "user", "content": prompt},
        ]
        self.history_messsages.clear()
        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=messages,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        response = response.choices[0].message.content
        self._save_cache(sys_prompt_arg + prompt, response)
        return response

    def _generate_from_multiple_input(
        self,
        content: List[Union[str, bytes]],
        system_prompt=None,
        temperature=0,
        max_tokens=2000,
        top_p=0.99,
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)

        cache_key = sys_prompt_arg + json.dumps(formatted_content)
        cache_or_none = self._check_cache(cache_key)
        if cache_or_none is not None:
            return cache_or_none

        messages = [
            {"role": "system", "content": sys_prompt_arg},
            *self.history_messsages,
            {"role": "user", "content": formatted_content},
        ]
        self.history_messsages.clear()
        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        response_text = response.choices[0].message.content
        self._save_cache(cache_key, response_text)
        return response_text


class BlackboxLLMWithHistory(tg.BlackboxLLM):
    def forward(self, x: Variable, history: list[dict] = []) -> Variable:
        if history and hasattr(self.engine, "inject_history"):
            self.engine.inject_history(history)

        return self.llm_call(x)


tg.set_backward_engine("gpt-4o", override=True)

# Step 1: Get an initial response from an LLM.
model = BlackboxLLMWithHistory(ChatOpenAIWithHistory("gpt-4o"))
question_string = (
    "If it takes 1 hour to dry 25 shirts under the sun, "
    "how long will it take to dry 30 shirts under the sun? "
    "Reason step by step"
)


question = tg.Variable(
    question_string, role_description="question to the LLM", requires_grad=False
)

history = [
    {
        "role": "user",
        "content": "Hi, how are you?",
    },
    {
        "role": "assistant",
        "content": "I'm fine!",
    },
]
answer = model(question, history=history)
print(answer)


answer.set_role_description("concise and accurate answer to the question")

# Step 2: Define the loss function and the optimizer, just like in PyTorch!
# Here, we don't have SGD, but we have TGD (Textual Gradient Descent)
# that works with "textual gradients".
optimizer = tg.TGD(parameters=[answer])
evaluation_instruction = (
    f"Here's a question: {question_string}. "
    "Evaluate any given answer to this question, "
    "be smart, logical, and very critical. "
    "Just provide concise feedback."
)


# TextLoss is a natural-language specified loss function that describes
# how we want to evaluate the reasoning.
loss_fn = tg.TextLoss(evaluation_instruction)

# Step 3: Do the loss computation, backward pass, and update the punchline.
# Exact same syntax as PyTorch!
loss = loss_fn(answer)
loss.backward()
optimizer.step()
answer1 = model(question, history)
print(answer1)