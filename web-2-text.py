#!/usr/bin/env python

from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from lxml.html.clean import clean_html
from pathlib import Path
from tqdm import tqdm
import itertools as it
import re
import os

URL_CHUNK_SIZE = 10
URL_FILE = os.getenv("URL_FILE")

text_file = open(URL_FILE, "r")
websites = text_file.read().splitlines()
websites_tuple = tuple(it.batched(websites, URL_CHUNK_SIZE))
pbar = tqdm(total=len(websites))

for x in websites_tuple:
    print(x)
    pbar.update(URL_CHUNK_SIZE)

    website_loader = WebBaseLoader(x)
    docs = website_loader.load()

    for d in docs:
        text = re.sub(r'^$\n', '', d.page_content, flags=re.MULTILINE)
        print(text)