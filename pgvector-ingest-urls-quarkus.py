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
import sys
import re

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

def process_websites(web_list, collection_name):
    print(f">> processing {web_list}")

    website_loader = WebBaseLoader(web_list)
    docs = website_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=40)
    all_splits = text_splitter.split_documents(docs)
    if len(all_splits) == 0:
        return
    all_splits[0]

    # Cleanup documents as PostgreSQL won't accept the NUL character, '\x00', in TEXT fields.
    for doc in all_splits:
        doc.page_content = doc.page_content.replace("\x00", "")

    # Create the index and ingest the documents
    embeddings = HuggingFaceEmbeddings(show_progress=True)

    db = PGVector.from_documents(
        documents=all_splits,
        embedding=embeddings,
        collection_name=collection_name,
        connection_string=CONNECTION_STRING,
        use_jsonb=True,
        # pre_delete_collection=True # This deletes existing collection and its data, use carefully!
    )

text_file = open(URL_FILE, "r")
websites = text_file.read().splitlines()
websites_tuple = tuple(it.batched(websites, URL_CHUNK_SIZE))
pbar = tqdm(total=len(websites))

for x in websites_tuple:
    print(x)
    pbar.update(URL_CHUNK_SIZE)

    store_docs_websites_list = list()
    store_main_websites_list = list()
    store_latest_websites_list = list()
    store_213_websites_list = list()
    store_216_websites_list = list()
    store_27_websites_list = list()
    store_38_websites_list = list()
    store_32_websites_list = list()

    for y in x:
        if check_duplicate(y):
            print(f">> skipping {y} already exists ...")
        else:
            if re.search(r'version/3.15', y, flags=re.IGNORECASE):
                store_latest_websites_list.append(y)
            elif re.search(r'http://0.0.0.0:4000/guides', y, flags=re.IGNORECASE):
                store_latest_websites_list.append(y)
            elif re.search(r'version/main', y, flags=re.IGNORECASE):
                store_main_websites_list.append(y)
            elif re.search(r'version/2.13', y, flags=re.IGNORECASE):
                store_213_websites_list.append(y)
            elif re.search(r'version/2.16', y, flags=re.IGNORECASE):
                store_216_websites_list.append(y)
            elif re.search(r'version/2.7', y, flags=re.IGNORECASE):
                store_27_websites_list.append(y)
            elif re.search(r'version/3.8', y, flags=re.IGNORECASE):
                store_38_websites_list.append(y)
            elif re.search(r'version/3.2', y, flags=re.IGNORECASE):
                store_32_websites_list.append(y)
            else:
                store_docs_websites_list.append(y)

    if len(store_latest_websites_list) != 0:
        process_websites(store_latest_websites_list, "latest")
    if len(store_main_websites_list) != 0:
        process_websites(store_main_websites_list, "main")
    if len(store_213_websites_list) != 0:
        process_websites(store_213_websites_list, "213")
    if len(store_216_websites_list) != 0:
        process_websites(store_216_websites_list, "216")
    if len(store_27_websites_list) != 0:
        process_websites(store_27_websites_list, "27")
    if len(store_38_websites_list) != 0:
        process_websites(store_38_websites_list, "38")
    if len(store_32_websites_list) != 0:
        process_websites(store_32_websites_list, "32")
    if len(store_docs_websites_list) != 0:
        process_websites(store_docs_websites_list, "docs")
