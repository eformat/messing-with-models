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

product_version = 2.12
CONNECTION_STRING = "postgresql+psycopg://postgres:password@localhost:5432/vectordb"
COLLECTION_NAME = "documents_test"

documents = [
    "release_notes",
    "introduction_to_red_hat_openshift_ai",
    "getting_started_with_red_hat_openshift_ai_self-managed",   
]

pdfs = [f"https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/{product_version}/pdf/{doc}/red_hat_openshift_ai_self-managed-{product_version}-{doc}-en-us.pdf" for doc in documents]
pdfs_to_urls = {f"red_hat_openshift_ai_self-managed-{product_version}-{doc}-en-us": f"https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/{product_version}/html-single/{doc}/index" for doc in documents}

try:
    os.mkdir(f"rhoai-doc-{product_version}")
except OSError as error:
    print(error)

for pdf in pdfs:
    try:
        response = requests.get(pdf)
    except:
        print(f"Skipped {pdf}")
        continue
    if response.status_code!=200:
        print(f"Skipped {pdf}")
        continue  
    with open(f"rhoai-doc-{product_version}/{pdf.split('/')[-1]}", 'wb') as f:
        f.write(response.content)

pdf_folder_path = f"./rhoai-doc-{product_version}"

pdf_loader = PyPDFDirectoryLoader(pdf_folder_path)
pdf_docs = pdf_loader.load()

# inject metadata
for doc in pdf_docs:
    doc.metadata["source"] = pdfs_to_urls[Path(doc.metadata["source"]).stem]

websites = [
    "https://ai-on-openshift.io/getting-started/openshift/",
    "https://ai-on-openshift.io/getting-started/opendatahub/",
    "https://ai-on-openshift.io/getting-started/openshift-ai/",
    "https://ai-on-openshift.io/odh-rhoai/configuration/",
    "https://ai-on-openshift.io/odh-rhoai/custom-notebooks/",
    "https://ai-on-openshift.io/odh-rhoai/nvidia-gpus/",
    "https://ai-on-openshift.io/odh-rhoai/custom-runtime-triton/",
    "https://ai-on-openshift.io/odh-rhoai/openshift-group-management/",
    "https://ai-on-openshift.io/tools-and-applications/minio/minio/",
    "https://blog.eformat.me/2024/07/vlans.html",
    "https://blog.eformat.me/2023/12/gitops-acm.html",
    "https://blog.eformat.me/2023/06/graphql-federation-quarkus.html",
    "https://blog.eformat.me/2023/05/smallrye-stork-load-balancing.html",
    "https://blog.eformat.me/2023/05/analytics.html",
    "https://blog.eformat.me/2023/04/disconnected-registries.html",
    "https://blog.eformat.me/2023/02/acm-team-argocd.html",
    "https://blog.eformat.me/2023/02/sno-metallb-bpg.html",
    "https://blog.eformat.me/2022/12/mastodon-openshift.html",
    "https://blog.eformat.me/2022/12/nvidia-gpu-sharing.html",
    "https://blog.eformat.me/2022/11/stable-diffusion.html",
    "https://blog.eformat.me/2022/11/aws-sno-150.html",
    "https://blog.eformat.me/2022/11/optaplanner-quarkus.html",
    "https://blog.eformat.me/2022/11/argocd-patterns-vault.html",
    "https://blog.eformat.me/2022/11/devops-with-openshift-5yr.html",
    "https://blog.eformat.me/2022/11/pulsar-flink.html",
    "https://blog.eformat.me/2022/11/the-compelling-platform.html"
]

website_loader = WebBaseLoader(websites)
website_docs = website_loader.load()

# merge
docs = pdf_docs + website_docs

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,
                                               chunk_overlap=40)
all_splits = text_splitter.split_documents(docs)
all_splits[0]

# Cleanup documents as PostgreSQL won't accept the NUL character, '\x00', in TEXT fields.
for doc in all_splits:
    doc.page_content = doc.page_content.replace('\x00', '')

# Create the index and ingest the documents
embeddings = HuggingFaceEmbeddings()

db = PGVector.from_documents(
    documents=all_splits,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    use_jsonb=True,
    #pre_delete_collection=True # This deletes existing collection and its data, use carefully!
)
