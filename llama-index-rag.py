#!/usr/bin/env python

# based on this example
# https://github.com/stanfordnlp/dspy/blob/main/examples/llamaindex/dspy_llamaindex_rag.ipynb

import dspy
dspy.configure(experimental=True) 
from llama_cpp import Llama

# set n_ctx and Settings.context_window for your model
llm = Llama(model_path="/opt/app-root/src/sno-llama/Meta-Llama-3-8B-Instruct-Q8_0.gguf", n_gpu_layers=-1, n_ctx=8192)

# llm model and retreival model
llamalm = dspy.LlamaCpp(model="llama", llama_model=llm,  model_type="chat", temperature=0.2, stop=('\\n',))
dspy.settings.configure(lm=llamalm)

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context_str = dspy.InputField(desc="contains relevant facts")
    query_str = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
    source = dspy.OutputField(desc="please cite sources")

# embedding model
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Load Data, Build Index

#wget https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt -O paul_graham_essay.txt

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext

reader = SimpleDirectoryReader(input_files=["paul_graham_essay.txt"])
docs = reader.load_data()

from llama_index.core import Settings
Settings.chunk_size=1024
Settings.context_window=8192
Settings.llm=llm
Settings.embed_model=embed_model
Settings.system_prompt=""

index = VectorStoreIndex.from_documents(docs)

retriever = index.as_retriever(similarity_top_k=2)

# Build Query Pipeline

from llama_index.core.query_pipeline import QueryPipeline as QP, InputComponent, FnComponent
from dspy.predict.llamaindex import DSPyComponent, LlamaIndexModule

dspy_component = DSPyComponent(
    dspy.Predict(GenerateAnswer)
)

retriever_post = FnComponent(
    lambda contexts: "\n\n".join([n.get_content() for n in contexts])
)

p = QP(verbose=True)
p.add_modules(
    {
        "input": InputComponent(),
        "retriever": retriever,
        "retriever_post": retriever_post,
        "synthesizer": dspy_component,
    }
)
p.add_link("input", "retriever")
p.add_link("retriever", "retriever_post")
p.add_link("input", "synthesizer", dest_key="query_str")
p.add_link("retriever_post", "synthesizer", dest_key="context_str")

dspy_qp = LlamaIndexModule(p)

output = dspy_qp(query_str="what did the author do in YC")

print(output)

# try optimizing with few shot examples

from dspy import Example

train_examples = [
    Example(query_str="What did the author do growing up?", answer="The author wrote short stories and also worked on programming."),
    Example(query_str="What did the author do during his time at YC?", answer="organizing a Summer Founders Program, funding startups, writing essays, working on a new version of Arc, creating Hacker News, and developing internal software for YC")
]

train_examples = [t.with_inputs("query_str") for t in train_examples]

import nest_asyncio
nest_asyncio.apply()

from dspy.teleprompt import BootstrapFewShot
from llama_index.core.evaluation import SemanticSimilarityEvaluator

evaluator = SemanticSimilarityEvaluator(similarity_threshold=0.5)

# Validation logic: check that the predicted answer is correct.
# Also check that the retrieved context does actually contain that answer.
def validate_context_and_answer(example, pred, trace=None):
    result = evaluator.evaluate(response=pred.answer, reference=example.answer)
    return result.passing

# Set up a basic teleprompter, which will compile our RAG program.
teleprompter = BootstrapFewShot(max_labeled_demos=0, metric=validate_context_and_answer)

# Compile!
compiled_dspy_qp = teleprompter.compile(dspy_qp, trainset=train_examples)

# test this out 
output = compiled_dspy_qp(query_str="How did PG meet Jessica Livingston?")

print(output)

