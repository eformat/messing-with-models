#!/usr/bin/env python

# psql -d vectordb -c "CREATE EXTENSION vector;"

from langchain_community.document_loaders import DirectoryLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from lxml.html.clean import clean_html
from pathlib import Path
import requests
import os

product_version = 2.12
CONNECTION_STRING = "postgresql+psycopg://postgres:password@localhost:5432/vectordb"
COLLECTION_NAME = "documents_test"

folder_path = f"/home/mike/tmp/ingest-markdown"
loader = DirectoryLoader(folder_path, glob="**/*.md")
m_docs = loader.load()

# inject metadata
for doc in m_docs:
    doc.metadata["source"] = doc.metadata["source"]

docs = m_docs

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=40)
all_splits = text_splitter.split_documents(docs)
all_splits[0]

# Cleanup documents as PostgreSQL won't accept the NUL character, '\x00', in TEXT fields.
for doc in all_splits:
    doc.page_content = doc.page_content.replace("\x00", "")

# Create the index and ingest the documents
embeddings = HuggingFaceEmbeddings()

db = PGVector.from_documents(
    documents=all_splits,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    use_jsonb=True,
    # pre_delete_collection=True # This deletes existing collection and its data, use carefully!
)
