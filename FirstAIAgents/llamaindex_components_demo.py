try:
    # Use consistent imports with llama-index-core
    from llama_index.core import SimpleDirectoryReader, Document, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.evaluation import FaithfulnessEvaluator
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
    from llama_index.core.vector_stores.chroma import ChromaVectorStore
except ImportError:
    # Fallback to older LlamaIndex structure
    from llama_index import SimpleDirectoryReader, Document, VectorStoreIndex
    from llama_index.node_parser import SentenceSplitter
    from llama_index.ingestion import IngestionPipeline
    from llama_index.evaluation import FaithfulnessEvaluator
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.huggingface import HuggingFaceInferenceAPI
    from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb
import asyncio
import os

# Uncomment and set your HuggingFace API token if using HuggingFace Inference API
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")
os.environ["HF_API_TOKEN"] = os.getenv("HF_API_TOKEN")

def setup_vector_store():
    """Initialize and return a ChromaDB vector store."""
    db = chromadb.PersistentClient(path="./llmdemo_chroma_db")
    chroma_collection = db.get_or_create_collection("llmdemo")
    return ChromaVectorStore(chroma_collection=chroma_collection)

def load_documents():
    """Load documents from a directory."""
    # For demo purposes, we'll create some sample documents
    documents = [
        Document(text="The American Revolution was a pivotal event in world history."),
        Document(text="The Battle of Long Island was fought in New York City in 1776."),
        Document(text="George Washington led the Continental Army during the Revolution."),
    ]
    return documents

def create_ingestion_pipeline(vector_store):
    """Create an ingestion pipeline with transformations."""
    return IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_overlap=0),
            HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
        ],
        vector_store=vector_store,
    )

async def process_and_index_documents(pipeline, documents):
    """Process documents through the ingestion pipeline and store in vector store."""
    print("Processing and indexing documents...")
    nodes = await pipeline.arun(documents=documents)
    print(f"Successfully indexed {len(nodes)} document chunks")
    return nodes

async def main():
    # 1. Setup vector store
    print("Setting up vector store...")
    vector_store = setup_vector_store()
    
    # 2. Load documents
    print("Loading documents...")
    documents = load_documents()
    
    # 3. Create ingestion pipeline
    print("Creating ingestion pipeline...")
    pipeline = create_ingestion_pipeline(vector_store)
    
    # 4. Process and index documents
    await process_and_index_documents(pipeline, documents)
    
    # 5. Create query engine
    print("Creating query engine...")
    # Get the LLM and embed model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    llm = HuggingFaceInferenceAPI(model_name="mistralai/Mistral-7B-Instruct-v0.2")
    
    # Create index and query engine with the LLM
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize",
    )
    
    # 6. Example queries
    print("\nQuerying the indexed documents...")
    queries = [
        "What battles took place in New York City during the American Revolution?",
        "Who led the Continental Army?",
        "When was the Battle of Long Island fought?"
    ]
    
    # Create evaluator once
    evaluator = FaithfulnessEvaluator(llm=llm)
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = await query_engine.aquery(query)
        print(f"Response: {response}")
        
        # Use aevaluate instead of evaluate_response to avoid nested asyncio loops
        eval_result = await evaluator.aevaluate(
            query=query,
            response=str(response),
            contexts=[str(n) for n in response.source_nodes]
        )
        print(f"Response faithfulness: {eval_result.passing}")

if __name__ == "__main__":
    asyncio.run(main()) 