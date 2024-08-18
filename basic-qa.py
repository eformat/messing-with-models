#!/usr/bin/env python

import dspy
dspy.configure(experimental=True) 
from llama_cpp import Llama

llm = Llama(model_path="/home/mike/instructlab/models/granite-7b-lab-Q4_K_M.gguf", n_gpu_layers=-1)

llamalm = dspy.LlamaCpp(model="llama", llama_model=llm,  model_type="chat", temperature=0.2, stop=('\\n',))
dspy.settings.configure(lm=llamalm)

# Define a simple signature for basic question answering
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# Pass signature to Predict module
generate_answer = dspy.Predict(BasicQA)

# Call the predictor on a particular input.
question='What is the color of the sky?'
pred = generate_answer(question=question)

print(f"Question: {question}")
print(f"Predicted Answer: {pred.answer}")
