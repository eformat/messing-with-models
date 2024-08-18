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
    answer = dspy.OutputField(desc="often between 1 and 20 words")

# embedding model
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Load Data, Build Index

#git clone https://github.com/rht-labs/tech-exercise.git

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.readers.file import MarkdownReader

parser = MarkdownReader()
file_extractor = {".md": parser}
docs = SimpleDirectoryReader(
    "./tech-exercise", file_extractor=file_extractor
).load_data()

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

query_str="what is devops?"
output = dspy_qp(query_str=query_str)
print(f"Question: {query_str}")
print(output)

query_str="how do i pratice everything as code?"
output = dspy_qp(query_str=query_str)
print(f"Question: {query_str}")
print(output)

query_str="what is definition of done?"
output = dspy_qp(query_str=query_str)
print(f"Question: {query_str}")
print(output)

query_str="how many cultural practices are there?"
output = dspy_qp(query_str=query_str)
print(f"Question: {query_str}")
print(output)

query_str="describe what TL500 is?"
output = dspy_qp(query_str=query_str)
print(f"Question: {query_str}")
print(output)
