"""To run:
python -m src.dspy.dspy
"""
import dspy
import os
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
lm = dspy.LM('openai/gpt-4o-mini', api_key=openai_api_key)
dspy.configure(lm=lm)


def initial_try():
    response_0 = lm("Say that you are DSPy and explain ur role", temperature=0.7)  
    print(response_0)
    response = lm(messages=[{"role": "user", "content": "Say this is a test!"}])  # => ['This is a test!']
    print(response)  # response is a list of strings in markdown format


def qa():
    # Define a module (ChainOfThought) and assign it a signature (return an answer, given a question).
    qa = dspy.ChainOfThought('question -> answer')

    # Run with the default LM configured with `dspy.configure` above.
    response = qa(question="How many floors are in the castle David Gregory inherited?")
    print(response.reasoning)
    print(response.answer)


class Emotion(dspy.Signature):
    """Classify emotion."""

    sentence: str = dspy.InputField()
    sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] = dspy.OutputField()


class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""

    context: str = dspy.InputField(desc="facts here are assumed to be true")
    text: str = dspy.InputField()
    faithfulness: bool = dspy.OutputField()
    evidence: dict[str, list[str]] = dspy.OutputField(desc="Supporting evidence for claims")


class DogPictureSignature(dspy.Signature):
    """Output the dog breed of the dog in the image."""
    image_1: dspy.Image = dspy.InputField(desc="An image of a dog")
    answer: str = dspy.OutputField(desc="The dog breed of the dog in the image")


def signature():
    """test signature of dspy"""
    sentence = "i started feeling a little vulnerable when the giant spotlight started blinding me"  # from dair-ai/emotion
    classify = dspy.Predict(Emotion)
    prediction = classify(sentence=sentence)
    print(prediction.sentiment)

    sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.
    classify = dspy.Predict('sentence -> sentiment: bool')  # we'll see an example with Literal[] later
    print(classify(sentence=sentence).sentiment)

    context = "The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."
    text = "Lee scored 3 goals for Colchester United."
    faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
    result = faithfulness(context=context, text=text)
    for key, value in result.items():
        print(key, value)
    # print(result)

    image_url = "https://picsum.photos/id/237/200/300"
    classify = dspy.Predict(DogPictureSignature)
    answer = classify(image_1=dspy.Image.from_url(image_url))
    print(answer)


def modules():
    question = "What's something great about the ColBERT retrieval model?"
    classify = dspy.ChainOfThought('question -> answer', n=5)
    response = classify(question=question)
    print(response.completions.answer)



if __name__ == "__main__":
    modules()