"""To run:
python -m src.dspy.multiturn_chat_adaptor

This is for run-timr conversation handling using DSPy
"""
import dspy
import os
from typing import Any
from dotenv import load_dotenv
load_dotenv()


def format_turn(signature, turn_content, role: str) -> dict[str, str]:
    """Format a conversation turn for the chat adapter."""
    content = ""
    for key, value in turn_content.items():
        if isinstance(value, dict):
            # Handle nested dictionary outputs
            for nested_key, nested_value in value.items():
                content += f"{nested_key}: {nested_value}\n"
        else:
            content += f"{key}: {value}\n"
    
    return {"role": role, "content": content.strip()}


class ChatModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("question -> answer")
        self.conversation_history = []
    
    def forward(self, question: str):
        prediction = self.predict(question=question, history_messages=self.conversation_history)
        self.conversation_history.append(
            {
                "inputs": dict(question=question),
                "outputs": prediction
            }
        )
        return prediction


class MultiTurnChatAdapter(dspy.ChatAdapter):
    def format(self, signature, demos, inputs) -> list[dict[str, Any]]:
        history_messages = inputs.pop("history_messages", [])
        inputs_ = super().format(signature, demos, inputs) # system prompt, demos, user prompt
        formatted_history = []
        for turn in history_messages:
            formatted_history.append(format_turn(signature, turn["inputs"], role="user"))
            formatted_history.append(format_turn(signature, turn["outputs"], role="assistant"))
        
        return inputs_[:-1] + formatted_history + [inputs_[-1]] # concat system and demos with past turns and finally the user prompt
    

# Example usage:
openai_api_key = os.getenv("OPENAI_API_KEY")
lm = dspy.LM('openai/gpt-4o-mini', api_key=openai_api_key)
dspy.configure(lm=lm)
dspy.settings.configure(adapter=MultiTurnChatAdapter())


def main():
    print("=== Simple conversation with memory ===")
    chat_module = ChatModule()
    
    # First message
    response1 = chat_module("My name is Itay")
    print("User: My name is Itay")
    print(f"Assistant: {response1.answer}")
    
    # Second message - should remember the name
    response2 = chat_module("What is my name?")
    print("User: What is my name?")
    print(f"Assistant: {response2.answer}")


if __name__ == "__main__":
    main()