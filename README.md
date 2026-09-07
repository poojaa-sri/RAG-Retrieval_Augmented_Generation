# Hands-On Notebook RAG

A hands-on **Retrieval-Augmented Generation (RAG)** project that combines document retrieval, semantic embeddings, FAISS vector search, and Llama 2 to generate context-grounded answers from company data.

## 📌 Project Overview

This project demonstrates how a RAG pipeline can retrieve relevant information from a document and provide accurate answers using a Large Language Model (LLM).

The project uses **Apple (AAPL) Management's Discussion and Analysis (MD&A)** data as the knowledge source. The document is divided into smaller chunks, converted into embeddings, stored in a FAISS vector database, and retrieved based on the user's query.

The retrieved context is then passed to **Llama 2 13B Chat** to generate the final answer.

## 🔄 RAG Pipeline

```text
Document
   ↓
Text Loading
   ↓
Text Chunking
   ↓
Sentence Transformer Embeddings
   ↓
FAISS Vector Store
   ↓
User Query
   ↓
Relevant Context Retrieval
   ↓
Llama 2 LLM
   ↓
Generated Answer
   ↓
Groundedness & Relevance Evaluation
```

## 🚀 Technologies Used

* Python
* LangChain
* Sentence Transformers
* BAAI BGE Base EN v1.5
* FAISS
* Llama 2 13B Chat
* LlamaCpp
* Hugging Face Hub
* PyTorch
* Google Colab

## 📂 Dataset

The project uses an **AAPL Management Discussion and Analysis (MD&A)** text document as the knowledge source.

```text
AAPL-MDA.txt
```

The document is loaded using LangChain's `TextLoader`.

## ⚙️ Implementation Steps

### 1. Install Required Libraries

The notebook installs the required packages including:

* `langchain`
* `langchain-community`
* `langchain-huggingface`
* `sentence-transformers`
* `faiss-cpu`
* `pypdf`
* `huggingface_hub`
* `llama-cpp-python`
* `torch`

### 2. Load the Document

The AAPL MD&A document is loaded and prepared for processing.

### 3. Text Chunking

The document is split into smaller chunks using `RecursiveCharacterTextSplitter`.

```python
chunk_size = 2000
chunk_overlap = 200
```

### 4. Generate Embeddings

The project uses:

```text
BAAI/bge-base-en-v1.5
```

to convert each text chunk into numerical vector representations.

### 5. Create FAISS Vector Store

FAISS is used to store the document embeddings and efficiently retrieve relevant chunks based on semantic similarity.

```python
vector_store = FAISS.from_documents(
    text_chunks,
    embedding=embeddings
)
```

### 6. Load Llama 2

The project downloads:

```text
Llama-2-13B-Chat GGUF
```

from Hugging Face and loads it using `LlamaCpp`.

### 7. Build the RAG Chain

A LangChain `RetrievalQA` chain is created with the Llama 2 model and FAISS retriever.

The retriever is configured to return the top **2 relevant chunks**.

### 8. Query the RAG System

Example query:

```text
How often does the company review inventory, and what is considered in this inventory calculation?
```

The system retrieves relevant information from the document and uses the retrieved context to generate an answer.

## 📊 Evaluation

The generated response is evaluated using two metrics:

### Relevance

Measures how well the generated answer addresses the original question.

**Rating:** 1–5

* 1 → Not relevant
* 2 → Slightly relevant
* 3 → Moderately relevant
* 4 → Mostly relevant
* 5 → Fully relevant

### Groundedness

Measures how well the generated answer is supported by the retrieved context.

**Rating:** 1–5

* 1 → Not derived from context
* 2 → Limited grounding
* 3 → Good grounding
* 4 → Mostly grounded
* 5 → Completely grounded

## 🎯 Key Learning Outcomes

* Understand the fundamentals of **Retrieval-Augmented Generation**
* Load and preprocess documents for RAG
* Split documents into meaningful text chunks
* Generate semantic embeddings using Sentence Transformers
* Store and retrieve embeddings using FAISS
* Integrate an LLM with a retrieval pipeline
* Build a RAG-based question-answering system using LangChain
* Evaluate LLM responses using **relevance and groundedness**

## 🌟 Project Highlights

* 📄 Document-based Question Answering
* 🔎 Semantic Search with FAISS
* 🧠 Llama 2 13B Chat
* 🔗 LangChain RAG Pipeline
* 📚 Sentence Transformer Embeddings
* 📊 Relevance Evaluation
* ✅ Groundedness Evaluation

## 👩‍💻 Author

**Poojasri Saravanakumar**

Aspiring AI & ML Engineer | Data Analytics & Generative AI Enthusiast
