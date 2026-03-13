#  Copyright (c) 2026 Schnitzer Christoph. All rights reserved.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import os
import torch

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCUMENT_DIR = os.getenv("DOCUMENT_DIR", "documents")
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")

# Multilingual embeddings model
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

def get_device():
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        print("No GPU detected, using CPU for embeddings")
        return "cpu"

def get_embeddings():
    device = os.getenv("EMBEDDING_DEVICE", get_device())
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 64 if device == "cuda" else 16,  # Larger batches on GPU
        }
    )

def ingest_documents():
    loaders = {
        "**/*.pdf": PyPDFLoader,
        "**/*.txt": TextLoader,
        "**/*.md": TextLoader
    }

    all_docs = []

    for pattern, loader_cls in loaders.items():
        loader = DirectoryLoader(
            DOCUMENT_DIR,
            glob=pattern,
            loader_cls=loader_cls,
            show_progress=True
        )
        all_docs.extend(loader.load())

    if not all_docs:
        print("No documents found. Add file to the documents folder.")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", ".", " ", ""]
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embeddings=embeddings,
        persist_directory=VECTORSTORE_DIR
    )
    print(f"Ingested {len(chunks)} chunks from {len(all_docs)} documents.")
    return vectorstore

def get_retriever(k: int = 4):
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})