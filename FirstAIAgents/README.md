# First AI Agents

This directory contains examples of AI agents built with LlamaIndex and Hugging Face models.

## Setup

### Create Python Virtual Environment
```bash
python -m venv venv
```

### Activate Python Environment
**Windows:**
```bash
.\venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### Install Dependencies
You can install dependencies using either method:

#### Option 1: Install from requirements.txt
```bash
pip install -r requirements.txt
```

#### Option 2: Install packages directly
```bash
pip install llama-index==0.10.12 llama-index-core==0.10.12 
pip install llama-index-llms-huggingface-api==0.4.1 llama-index-embeddings-huggingface==0.4.1
pip install chromadb==0.4.22 sentence-transformers==2.6.1
```

### Hugging Face Authentication
```bash
huggingface-cli login --token "your-token"
```
Alternatively, set environment variables:
```bash
# Windows
set HF_TOKEN=your-token
set HUGGINGFACE_API_KEY=your-token

# Linux/macOS
export HF_TOKEN=your-token
export HUGGINGFACE_API_KEY=your-token
```

## Examples

### Simple LLM Agent
The [simpleLLMAgent.py](./simpleLLMAgent.py) demonstrates a basic interaction with a Hugging Face model.

### LlamaIndex Components Demo
The [llamaindex_components_demo.py](./llamaindex_components_demo.py) shows a complete RAG (Retrieval Augmented Generation) workflow with:
- Document loading and processing
- Sentence splitting and embedding
- Vector storage with ChromaDB
- Query processing with LLMs
- Response evaluation