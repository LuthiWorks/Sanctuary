"""
ChromaDB-compatible embedding function wrapper for sentence-transformers.

Wraps HuggingFace sentence-transformers to work with ChromaDB's new API.
"""

from typing import List
import logging
from chromadb import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger(__name__)


class ChromaCompatibleEmbeddings(EmbeddingFunction):
    """
    ChromaDB-compatible embedding function using sentence-transformers.
    
    Implements the new ChromaDB EmbeddingFunction interface:
    - __call__(self, input: Documents) -> Embeddings
    
    This replaces the old LangChain HuggingFaceEmbeddings which used
    the signature (*args, **kwargs) that is no longer compatible.
    """
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32
    ):
        """
        Initialize the embedding function.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run on ('cuda', 'cpu', or None for auto)
            normalize_embeddings: Whether to normalize embeddings
            batch_size: Maximum batch size for encoding (prevents OOM)
        """
        if not model_name:
            raise ValueError("model_name cannot be empty")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        
        # Determine device
        if device is None:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Load model with error handling
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence-transformers model: {model_name} on {device}")
            self.model = SentenceTransformer(model_name, device=device)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Could not initialize embedding model '{model_name}': {e}") from e
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Encode documents into embeddings.
        
        Args:
            input: List of document strings to embed
            
        Returns:
            List of embedding vectors (list of lists of floats)
            
        Raises:
            ValueError: If input is empty or contains invalid types
            RuntimeError: If encoding fails
        """
        if not isinstance(input, (list, tuple)):
            raise TypeError(f"Input must be list or tuple, got {type(input).__name__}")

        if len(input) == 0:
            # Fail loudly. This used to warn and return [], which the
            # docstring above already contradicted -- an embedding call that
            # silently produces nothing is indistinguishable downstream from
            # one that produced nothing because the model was broken. A caller
            # asking to embed an empty batch has a bug; say so here rather
            # than let it surface as an empty ChromaDB write later.
            raise ValueError("Cannot embed an empty document list")
        
        # Validate all inputs are strings
        for i, doc in enumerate(input):
            if not isinstance(doc, str):
                raise TypeError(f"All documents must be strings, item {i} is {type(doc).__name__}")
        
        try:
            # Encode texts to embeddings with batch size limit
            embeddings = self.model.encode(
                input,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=self.batch_size  # Prevent OOM on large batches
            )
            
            # ChromaDB Embeddings type is List[np.ndarray]
            return list(embeddings)
            
        except Exception as e:
            logger.error(f"Failed to encode {len(input)} documents: {e}")
            raise RuntimeError(f"Embedding encoding failed: {e}") from e
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        LangChain-compatible method for embedding documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self.__call__(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        LangChain-compatible method for embedding a single query.
        
        Args:
            text: Single text to embed
            
        Returns:
            Single embedding vector
        """
        return self.__call__([text])[0]
