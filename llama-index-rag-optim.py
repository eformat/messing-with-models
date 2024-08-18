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

# embedding model
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

from llama_index.core import Settings
Settings.chunk_size=1024
Settings.context_window=8192
Settings.llm=llm
Settings.embed_model=embed_model

# Build and Optimize Query Pipelines with Existing Prompts
from llama_index.core.prompts import PromptTemplate

# let's try a fun prompt that writes in Shakespeare! 
qa_prompt_template = PromptTemplate("""\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, \
answer the query.

Write in the style of a Shakespearean sonnet.

Query: {query_str}
Answer: 
""")

from llama_index.core.query_pipeline import QueryPipeline as QP, InputComponent, FnComponent
from dspy.predict.llamaindex import DSPyComponent, LlamaIndexModule

dspy_component = DSPyComponent.from_prompt(qa_prompt_template)

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

# check the inferred signature
dspy_component.predict_module.signature
