"""To run:
python -m src.dspy.dspy_optimizer
Reference: 
https://github.com/Pavankunchala/LLM-Learn-PK/blob/main/DSP/Medical_bot/doctor_app_BRandom.py
https://github.com/Pavankunchala/LLM-Learn-PK/blob/main/DSP/Medical_bot/Medical_BRandom.py
Optimizor: https://dspy.ai/learn/optimization/optimizers/?h=optimizer
"""
import dspy
import os
from dotenv import load_dotenv
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dspy.teleprompt import *
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
lm = dspy.LM('openai/gpt-4o-mini', api_key=openai_api_key)
dspy.configure(lm=lm)


# Create a dataset of examples
def create_conversation_dataset():
    examples = [
        # Example 1 - Course inquiry
        dspy.Example(
            conversation=[
                {"role": "user", "content": "i want to know about courses"},
                {"role": "assistant", "content": "may i get your kid's age and what kind of course is he or she interested in?"},
                {"role": "user", "content": "my kid is 9 years old, interested in english reading"},
                {"role": "assistant", "content": "there is a english reading course for ur kid, every tuesday and thursday, 7pm to 8pm"},
            ]
        ).with_inputs("conversation"),
        
        # Example 2 - Swimming lessons
        dspy.Example(
            conversation=[
                {"role": "user", "content": "do you offer swimming lessons?"},
                {"role": "assistant", "content": "yes, we do. may i know the age of the person interested and their swimming experience?"},
                {"role": "user", "content": "my daughter is 7 with no prior experience"},
                {"role": "assistant", "content": "we have beginner swimming classes for 6-8 year olds on mondays and wednesdays from 4pm to 5pm"},
            ]
        ).with_inputs("conversation"),
        
        # Example 3 - Teen programs
        dspy.Example(
            conversation=[
                {"role": "user", "content": "what programs do you have for teenagers?"},
                {"role": "assistant", "content": "we have several programs for teenagers including sports, arts, and academic enrichment. what specific interests does the teenager have?"},
                {"role": "user", "content": "she's interested in technology and coding"},
                {"role": "assistant", "content": "we offer coding classes for teens on tuesdays and thursdays from 4-6pm. would you like to know about beginner or advanced options?"},
                {"role": "user", "content": "what are the beginner options?"},
                {"role": "assistant", "content": "our beginner coding class covers Python basics, web development fundamentals, and simple game creation. it's suitable for ages 13-17 with no prior experience and includes all necessary equipment."},
            ]
        ).with_inputs("conversation"),
        
        # Example 4 - Music lessons
        dspy.Example(
            conversation=[
                {"role": "user", "content": "Do you have piano lessons?"},
                {"role": "assistant", "content": "Yes, we offer piano lessons for various age groups and skill levels. Could you tell me the age of the student and their current piano experience?"},
                {"role": "user", "content": "My son is 12 and has been playing for about 2 years"},
                {"role": "assistant", "content": "For a 12-year-old with 2 years of experience, we recommend our Intermediate Piano program. Classes are on Mondays and Thursdays from 5-6pm with instructor Ms. Johnson. Students work on classical and contemporary pieces while advancing their technical skills."},
            ]
        ).with_inputs("conversation"),
        
        # Example 5 - Sports inquiry
        dspy.Example(
            conversation=[
                {"role": "user", "content": "Do you have any soccer programs?"},
                {"role": "assistant", "content": "Yes, we offer soccer programs for different age groups. What's the age of the person interested in soccer?"},
                {"role": "user", "content": "My twins are 8 years old"},
                {"role": "assistant", "content": "We have a youth soccer program for 7-9 year olds that meets every Tuesday and Saturday morning from 9-10:30am. The program focuses on developing fundamental skills through fun drills and small-sided games. All equipment is provided, but players should bring their own water bottle."},
            ]
        ).with_inputs("conversation"),
        
        # Example 6 - Adult education
        dspy.Example(
            conversation=[
                {"role": "user", "content": "Do you have any classes for adults?"},
                {"role": "assistant", "content": "Yes, we offer various adult education programs. What subjects or activities are you interested in?"},
                {"role": "user", "content": "I'd like to learn photography"},
                {"role": "assistant", "content": "We offer a Digital Photography course for adults on Wednesday evenings from 6:30-8:30pm. The 8-week course covers camera basics, composition, lighting, and editing. It's suitable for beginners and intermediate photographers. The next session starts on June 5th and costs $195, which includes all materials except your camera."},
            ]
        ).with_inputs("conversation"),
        
        # Example 7 - Camp schedule
        dspy.Example(
            conversation=[
                {"role": "user", "content": "When do your summer camps start?"},
                {"role": "assistant", "content": "Our summer camps run from June 15 through August 20. Are you looking for a specific age group or type of camp?"},
                {"role": "user", "content": "Science camp for elementary school"},
                {"role": "assistant", "content": "Our Science Explorers camp for elementary students (ages 6-11) has three sessions: June 15-26, July 13-24, and August 3-14. Each runs weekdays from 9am-3pm with options for extended care. Activities include experiments, robotics, nature exploration, and STEM challenges. Registration is $325 per two-week session with a 10% sibling discount."},
            ]
        ).with_inputs("conversation")
    ]
    return examples


# Define a signature for conversation continuation
class ConversationContinuation(dspy.Signature):
    """Continue a conversation by generating the next assistant response."""
    conversation = dspy.InputField(desc="The conversation history as a list of messages")
    response = dspy.OutputField(desc="The next assistant response")


# Implement semantic similarity metrics similar to DeepEval's approach
class OpenAISemanticEvaluator:
    """Uses OpenAI embeddings for semantic similarity computation"""
    
    def __init__(self, model="text-embedding-3-small"):
        self.model = model
        self.client = openai.OpenAI(api_key=openai_api_key)
    
    def get_embedding(self, text):
        """Get OpenAI embedding for a text string"""
        if not text.strip():
            return np.zeros(1536)  # Default dimension for text-embedding-3-small
            
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return np.array(response.data[0].embedding)
    
    def compute_similarity(self, text1, text2):
        """Compute cosine similarity between two texts"""
        # Get embeddings
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        
        # Compute cosine similarity
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        return similarity


# Define LLM-based evaluation for conversation quality
class ConversationalGEval:
    """Similar to DeepEval's ConversationalGEval metric"""
    def __init__(self, criteria, threshold=0.7):
        self.criteria = criteria
        self.threshold = threshold
    
    def evaluate(self, example, prediction):
        # Extract the conversation history
        conv_history = example.conversation
        
        # Get the last assistant message from gold standard
        last_assistant_msgs = [msg["content"] for msg in conv_history if msg["role"] == "assistant"]
        if not last_assistant_msgs:
            return 0.0
        
        gold_response = last_assistant_msgs[-1]
        pred_response = prediction.response
        
        # Use LLM to evaluate the quality
        evaluation_prompt = f"""
        Evaluate the quality of a predicted response compared to a ground truth response.
        
        Criteria: {self.criteria}
        
        Ground Truth: {gold_response}
        
        Predicted: {pred_response}
        
        Rate the similarity on a scale of 0.0 to 1.0, where:
        - 0.0 means completely different or incorrect
        - 1.0 means semantically equivalent or perfectly appropriate
        
        Output only a single float number between 0.0 and 1.0.
        """
        
        # Run evaluation through the LM
        with dspy.context(lm=lm):
            result = dspy.Predict("prompt -> score")(prompt=evaluation_prompt)
            try:
                score = float(result.score)
                return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
            except ValueError:
                # Fallback to OpenAI embeddings if LLM doesn't return a valid float
                semantic_evaluator = OpenAISemanticEvaluator()
                return semantic_evaluator.compute_similarity(gold_response, pred_response)


# Define a comprehensive metric that combines different aspects
def comprehensive_conversation_metric(example, prediction, trace=None):
    """Evaluates multiple dimensions of conversation quality"""
    
    # Initialize different evaluators
    semantic_evaluator = OpenAISemanticEvaluator()
    
    # Extract last gold standard response and predicted response
    conv_history = example.conversation
    last_assistant_msgs = [msg["content"] for msg in conv_history if msg["role"] == "assistant"]
    if not last_assistant_msgs:
        return 0.0
        
    gold_response = last_assistant_msgs[-1]
    pred_response = prediction.response
    
    # Calculate different metrics
    semantic_similarity = semantic_evaluator.compute_similarity(gold_response, pred_response)
    
    # For optimization phase, simply check if it meets a threshold
    if trace is not None:
        return semantic_similarity > 0.7
    
    # For evaluation phase, return the semantic similarity score
    return semantic_similarity


class ConversationBot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(ConversationContinuation)
    
    def forward(self, conversation):
        # Extract the conversation up to the last user message
        user_turns = sum(1 for msg in conversation if msg["role"] == "user")
        assistant_turns = sum(1 for msg in conversation if msg["role"] == "assistant")
        
        # If this is a training example with complete conversation
        if assistant_turns >= user_turns:
            input_conv = conversation[:-1]  # Remove last assistant message
        else:
            input_conv = conversation
            
        return self.generate(conversation=input_conv)


def train_conversation_model(model_path="./model/conversation_model.json"):
    dataset = create_conversation_dataset()
    # Check if model already exists
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model = ConversationBot()
        model.load(model_path)
        return model
    
    # Create the model
    model = ConversationBot()
    
    # Split dataset
    train_data = dataset[:5]  # First two examples for training
    test_data = dataset[5:]   # Last example for testing
    
    # Initialize a conversational evaluator with specific criteria
    conversational_evaluator = ConversationalGEval(
        criteria="""
        Evaluate on these dimensions:
        1. Relevance: Response directly addresses user's question/request
        2. Appropriateness: Response is helpful and professional
        3. Completeness: Response provides all necessary information
        4. Role Adherence: Response maintains appropriate role and tone
        5. Knowledge Retention: Response correctly references information from earlier in conversation
        """,
        threshold=0.7
    )
    
    # Custom metric function that uses the conversational evaluator
    def geval_metric(example, pred, trace=None):
        score = conversational_evaluator.evaluate(example, pred)
        if trace is not None:
            return score >= conversational_evaluator.threshold
        return score
    
    # Optimize using MIPROv2 with our custom metric
    optimizer = dspy.MIPROv2(metric=geval_metric)
    # optimized_model = optimizer.optimize(model, train_data=train_data, test_data=test_data)
    optimized_model = optimizer.compile(
        student=model,
        trainset=train_data,
        valset=test_data,
        minibatch_size=1,  # Set minibatch size to 1
        minibatch=True     # Enable minibatching
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"Saving optimized model to {model_path}...")
    optimized_model.save(model_path)
    return optimized_model


def main():
    model = train_conversation_model()
    
    # Test with new conversation - 2 turns
    conversation_2turn = [
        {"role": "user", "content": "I'd like information about art classes"},
        {"role": "assistant", "content": "Great! I'd be happy to help you find art classes. Could you tell me who the classes are for and what type of art they're interested in?"},
        {"role": "user", "content": "for my 10-year old who enjoys painting"},
    ]
    
    result = model(conversation=conversation_2turn)
    print("\n=== 2-Turn Conversation ===")
    for msg in conversation_2turn:
        print(f"{msg['role'].capitalize()}: {msg['content']}")
    print(f"Assistant: {result.response}")
    
    # Test with new conversation - 3 turns
    conversation_3turn = [
        {"role": "user", "content": "Do you have summer camps?"},
        {"role": "assistant", "content": "Yes, we offer various summer camps for different age groups and interests. Could you tell me who would be attending and what they might be interested in?"},
        {"role": "user", "content": "I'm looking for something for my teenager"},
        {"role": "assistant", "content": "We have several teen summer camps including sports, technology, arts, and leadership programs. What specific interests does your teenager have?"},
        {"role": "user", "content": "He's interested in outdoor activities"},
    ]
    
    result = model(conversation=conversation_3turn)
    print("\n=== 3-Turn Conversation ===")
    for msg in conversation_3turn:
        print(f"{msg['role'].capitalize()}: {msg['content']}")
    print(f"Assistant: {result.response}")


if __name__ == "__main__":
    main()