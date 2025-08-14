#!/usr/bin/env python

# psql -d vectordb -c "CREATE EXTENSION vector;"

from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from lxml.html.clean import clean_html
from pathlib import Path
from tqdm import tqdm
import itertools as it
import requests
import psycopg
import os

COLLECTION_NAME = "documents_test"
DB_HOST = os.getenv("DB_HOST", "host=localhost")
DB_PORT = os.getenv("DB_PORT", "port=5432")
DB_NAME = os.getenv("DB_NAME", "dbname=vectordb")
DB_USER = os.getenv("DB_USER", "user=postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password=password")
CONNECTION_STRING = os.getenv("CONNECTION_STRING", "postgresql+psycopg://postgres:password@localhost:5432/vectordb")
URL_CHUNK_SIZE = 10
URL_FILE = os.getenv("URL_FILE")
conn = psycopg.connect("%s %s %s %s %s" % (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD))

def check_duplicate(uri):
    with conn.cursor() as cursor:
        try:
            cursor.execute(
                "select distinct cmetadata->>'source' as source from langchain_pg_embedding where cmetadata->>'source' = '%s'"
                % uri
            )
        except psycopg.errors.UndefinedTable as e:
            return False
        rows = cursor.fetchone()
        if not rows:
            return False
        for row in rows:
            return True

text_file = open(URL_FILE, "r")
websites = text_file.read().splitlines()
websites_tuple = tuple(it.batched(websites, URL_CHUNK_SIZE))
pbar = tqdm(total=len(websites))

headers = {
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9',
}

for x in websites_tuple:
    print(x)
    pbar.update(URL_CHUNK_SIZE)

    websites_list = list()

    for y in x:
        if check_duplicate(y):
            print(f">> skipping {y} already exists ...")
        else:
            websites_list.append(y)

    if len(websites_list) == 0:
        continue
    #print(f">> processing {websites_list}")

    website_loader = WebBaseLoader(websites_list, header_template=headers)
    docs = website_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=40)
    all_splits = text_splitter.split_documents(docs)
    if len(all_splits) == 0:
        continue
    all_splits[0]

    # Cleanup documents as PostgreSQL won't accept the NUL character, '\x00', in TEXT fields.
    for doc in all_splits:
        doc.page_content = doc.page_content.replace("\x00", "")

    # Create the index and ingest the documents
    embeddings = HuggingFaceEmbeddings(show_progress=True)

    db = PGVector.from_documents(
        documents=all_splits,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        use_jsonb=True,
        # pre_delete_collection=True # This deletes existing collection and its data, use carefully!
    )
