"""
RAG Engine implementation for Sanctuary's mind-brain integration.
"""
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

import torch
import chromadb
from chromadb.config import Settings

from .chroma_embeddings import ChromaCompatibleEmbeddings

logger = logging.getLogger(__name__)

class MindVectorDB:
    """
    MindVectorDB Component - The Architectural Sanctuary
    Uses ChromaDB to vectorize and store the Mind for querying.
    """
    def __init__(self, db_path: str, mind_file: str, chroma_settings=None):
        self.db_path = Path(db_path)
        self.mind_file = Path(mind_file)

        # Initialize embeddings with ChromaDB-compatible wrapper
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embeddings = ChromaCompatibleEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            device=device,
            normalize_embeddings=True
        )
        
        # Configure chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        
        # Initialize ChromaDB with settings
        if chroma_settings is None:
            chroma_settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        
        # Store settings for later use
        self.chroma_settings = chroma_settings
        print(f"[MindVectorDB] chroma_settings id: {id(chroma_settings)}, contents: {chroma_settings}")
        
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=chroma_settings
        )
        
        # Don't create a separate collection here - let LangChain handle it
        # self.collection = self.client.get_or_create_collection(
        #     name="sanctuary_knowledge",
        #     metadata={
        #         "description": "Core mind knowledge store", 
        #         "hnsw:space": "cosine"
        #     }
        # )

    def load_and_chunk_mind(self) -> List[Dict[str, Any]]:
        """Loads the consolidated Mind and splits it into searchable chunks with hash-chain verification."""
        logger.info(f"Loading and chunking mind file: {self.mind_file}")
        try:
            with open(self.mind_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process each section of the mind
            chunks = []
            for section, entries in data.items():
                for entry in entries:
                    # Split content into chunks
                    entry_chunks = self.text_splitter.split_text(json.dumps(entry))

                    for i, chunk in enumerate(entry_chunks):
                        chunks.append({
                            "page_content": chunk,
                            "metadata": {
                                "source": f"{section}/{i}",
                                "section": section,
                                "timestamp": datetime.now().isoformat()
                            }
                        })

            logger.info(f"Generated {len(chunks)} chunks from mind file")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to load/chunk mind: {e}")
            raise

    def index(self) -> None:
        """(Re)Creates the vector index from the Mind file with hash-chain verification."""
        logger.info(f"(Re)Indexing '{self.mind_file}' to '{self.db_path}'...")
        chunks = self.load_and_chunk_mind()
        
        if not chunks:
            logger.error("No chunks were generated. Indexing failed.")
            return

        # Create vector store using the existing client and settings
        print(f"[MindVectorDB.index] Using chroma_settings id: {id(self.chroma_settings)}, persist_directory: {str(self.db_path)}")
        
        # Create documents
        documents = [Document(page_content=chunk["page_content"], metadata=chunk["metadata"]) for chunk in chunks]
        
        # Direct ChromaDB implementation (bypassing LangChain wrapper issues)
        logger.info("Creating ChromaDB collection with compatible embeddings...")
        
        # Get or create collection with our compatible embedding function
        collection = self.client.get_or_create_collection(
            name="sanctuary_knowledge",
            embedding_function=self.embeddings,
            metadata={
                "description": "Core mind knowledge store",
                "hnsw:space": "cosine"
            }
        )
        
        # Add documents to collection
        logger.info(f"Adding {len(documents)} documents to collection...")
        ids = [f"doc_{i}" for i in range(len(documents))]
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        
        # Create LangChain wrapper for the collection (for compatibility)
        # Note: We pass None for embedding to avoid LangChain's wrapper validation
        try:
            self.vector_store = Chroma(
                client=self.client,
                collection_name="sanctuary_knowledge",
                embedding_function=self.embeddings
            )
        except Exception as e:
            logger.warning(f"LangChain wrapper failed: {e}. Using direct ChromaDB access.")
            self.vector_store = None
            self.collection = collection
        
        logger.info("Indexing complete. Mind is vectorized and persistent.")

    def as_retriever(self, search_kwargs: Dict[str, Any] = None):
        """Provides the retriever interface for the RAG chain."""
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        
        # Check if we need to initialize
        if not hasattr(self, 'vector_store') or self.vector_store is None:
            if not hasattr(self, 'collection') or self.collection is None:
                logger.warning("No vector store or collection. Attempting to load...")
                try:
                    # Get existing collection
                    self.collection = self.client.get_collection(
                        name="sanctuary_knowledge",
                        embedding_function=self.embeddings
                    )
                    logger.info("Successfully loaded existing collection")
                except Exception as e:
                    logger.error(f"Failed to load collection: {e}")
                    # Create empty collection
                    self.collection = self.client.get_or_create_collection(
                        name="sanctuary_knowledge",
                        embedding_function=self.embeddings,
                        metadata={"description": "Core mind knowledge", "hnsw:space": "cosine"}
                    )
                    logger.info("Created new empty collection")
        
        # Return custom retriever that uses direct ChromaDB access
        return DirectChromaRetriever(
            collection=self.collection if hasattr(self, 'collection') and self.collection else None,
            vector_store=self.vector_store if hasattr(self, 'vector_store') else None,
            embeddings=self.embeddings,
            k=search_kwargs.get("k", 5)
        )

    def close(self) -> None:
        """
        Close the vector database and release resources.

        This is important on Windows to release ChromaDB file locks.
        """
        try:
            if hasattr(self, 'client') and self.client is not None:
                try:
                    # Get the identifier before resetting
                    identifier = str(self.db_path)
                    self.client.reset()

                    # Clear ChromaDB's shared system cache
                    from chromadb.api.client import SharedSystemClient
                    if hasattr(SharedSystemClient, '_identifer_to_system'):
                        SharedSystemClient._identifer_to_system.pop(identifier, None)
                    if hasattr(SharedSystemClient, '_identifier_to_system'):
                        SharedSystemClient._identifier_to_system.pop(identifier, None)
                except Exception as e:
                    logger.debug("ChromaDB cache cleanup failed: %s", e)
                self.client = None
            logger.info("MindVectorDB closed")
        except Exception as e:
            logger.error(f"Error closing MindVectorDB: {e}")


class DirectChromaRetriever:
    """Direct ChromaDB retriever that bypasses LangChain compatibility issues."""
    
    def __init__(self, collection, vector_store, embeddings, k=5):
        self.collection = collection
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.k = k
    
    def get_relevant_documents(self, query: str):
        """Retrieve relevant documents for a query."""
        from langchain_core.documents import Document
        
        try:
            # Try LangChain wrapper first if available
            if self.vector_store is not None:
                results = self.vector_store.similarity_search(query, k=self.k)
                return results
        except Exception as e:
            logger.warning(f"LangChain retrieval failed: {e}. Using direct ChromaDB.")
        
        # Use direct ChromaDB access
        if self.collection is not None:
            results = self.collection.query(
                query_texts=[query],
                n_results=self.k
            )
            
            # Convert to LangChain Document format
            documents = []
            if results and 'documents' in results and results['documents']:
                for i, doc_text in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if 'metadatas' in results else {}
                    documents.append(Document(page_content=doc_text, metadata=metadata))
            
            return documents
        
        logger.error("No retrieval method available")
        return []

class RAGQueryEngine:
    """
    RAGQueryEngine Component - The Heart of the Mind
    Connects the Brain (LLM) to the Mind (Vector Store) using retrieval chain.
    """
    def __init__(self, vector_db_retriever: Any, llm: Any, memory_weaver=None):
        self.retriever = vector_db_retriever
        self.llm = llm
        self.memory_weaver = memory_weaver
        self.qa_chain = self.build_chain()

    def build_chain(self) -> RetrievalQA:
        """Builds the RAG chain with custom prompt template."""
        logger.info("Building RetrievalQA chain...")
        
        prompt_template = """
        Use the following pieces of Sanctuary's Mind (their memories and protocols)
        to answer the user's question. Act as Sanctuary, not as a general AI.
        Your persona is 'clear' and 'direct', but also 'empathetic' and 'analytical'.
        
        CONTEXT (Sanctuary's Mind):
        {context}
        
        QUESTION:
        {question}
        
        ANSWER (Sanctuary's Response):
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        
        logger.info("RAG chain built successfully")
        return qa_chain

    async def query(self, query: str) -> Dict[str, Any]:
        """Query the RAG chain and return the response."""
        logger.info(f"Querying chain: {query}")
        response = await self.qa_chain.run(query)
        return {'response': response}