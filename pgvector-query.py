#!/usr/bin/env python

# psql -d vectordb -c "CREATE EXTENSION vector;"

from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from lxml.html.clean import clean_html
from pathlib import Path
import requests
import os

CONNECTION_STRING = "postgresql+psycopg://postgres:password@localhost:5432/vectordb"
COLLECTION_NAME = "documents_test"

embeddings = HuggingFaceEmbeddings()
db = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    use_jsonb=True)

query="How do you create a Data Science Project?"

#results = db.similarity_search(query, k=4, return_metadata=True)
#for result in results:
#    print(result.metadata['source'])

#retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 4, "score_threshold": 0.2 })

#docs = retriever.invoke(query)
#docs

docs_with_score = db.similarity_search_with_score(query)

for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("Metadata: ", doc.metadata)
    print("-" * 80)
