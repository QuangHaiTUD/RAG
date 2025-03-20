# Retrieval-Augmented Generation (RAG) Model

## Overview
This project implements a Retrieval-Augmented Generation (RAG) model to answer queries based on a combination of information retrieval and text generation techniques. It uses `langchain`, `transformers`, `FAISS`, and other powerful libraries to provide accurate and contextually relevant responses. The vector database is stored in Google Drive, and embeddings are generated using pre-trained models.

---

## Features
- **Document Loading**: Supports PDF and directory loaders for extracting data.
- **Text Splitting**: Uses character-based splitting to preprocess large documents.
- **Vector Store**: Implements FAISS for efficient similarity search.
- **Embeddings**: Generates sentence embeddings with `sentence-transformers`.
- **RAG Pipeline**: Combines information retrieval and language generation using the `langchain` framework.
- **Evaluation**: Measures similarity scores using cosine similarity.

---

## Requirements
The following libraries and tools are required:
- Python 3.7+
- `langchain`
- `transformers`
- `sentence-transformers`
- `faiss-cpu`
- `torch`
- `pypdf`
- `chainlit`

---

## Installation
1. Install the required libraries:
   ```bash
   pip install langchain transformers sentence-transformers faiss-cpu torch pypdf chainlit
   ```
2. If using Google Colab:
   - Mount Google Drive to save/load vector databases:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```

---

## Usage
### 1. Document Preprocessing
- Load PDF or directory data:
  ```python
  from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
  loader = PyPDFLoader("path_to_pdf")
  documents = loader.load()
  ```
- Split text into manageable chunks:
  ```python
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  splitter = RecursiveCharacterTextSplitter()
  chunks = splitter.split_documents(documents)
  ```

### 2. Embedding and Vector Store
- Generate embeddings:
  ```python
  from langchain_community.embeddings import HuggingFaceEmbeddings
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  ```
- Save to FAISS vector store:
  ```python
  from langchain_community.vectorstores import FAISS
  vectorstore = FAISS.from_documents(chunks, embeddings)
  vectorstore.save_local("/path/to/vectorstore")
  ```

### 3. Retrieval-Augmented Generation
- Load vector store:
  ```python
  vectorstore = FAISS.load_local("/path/to/vectorstore", embeddings)
  ```
- Build RAG pipeline:
  ```python
  from langchain.chains import RetrievalQA
  rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
  ```
- Query the model:
  ```python
  response = rag_chain.run({"query": "Your question here"})
  print(response)
  ```

---

## Example
1. Load a PDF file.
2. Split it into chunks and generate embeddings.
3. Save the embeddings into FAISS.
4. Query the model using the RAG pipeline to get contextually relevant answers.

---

## Limitations
- Requires pre-trained embeddings and a properly configured vector store.
- Dependent on the quality of the input documents for retrieval accuracy.

---

## Contact
For further questions or issues, please contact:
- **Name**: Dang Quang Hai
- **Email**: dangquanghai.word@gmail.com

