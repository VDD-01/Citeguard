"""Dense embedding module for CiteGuard-RAG.

Wraps sentence-transformers to provide normalized dense vector representations
for document chunks and queries, supporting cosine similarity computation and
persistent disk-level embedding caches.

Matches Section 2.3 of Barua et al. (2026):
c_i = f(C_i),  q = f(q)
Sim_semantic(q, C_i) = (q · c_i) / (||q|| ||c_i||)
Default model: BAAI/bge-base-en-v1.5
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import hashlib
import numpy as np

from src.citeguard.config import EmbeddingConfig, load_config
from src.citeguard.models.chunk import DocumentChunk
from src.citeguard.utils.logging import setup_logger

logger = setup_logger(__name__)


class EmbeddingModel:
    """Singleton-style wrapper around sentence-transformers models.

    Ensures the heavy transformer model is loaded exactly once, encodes texts in
    batches, produces unit-normalized vectors, and caches representations on disk.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cpu",
        normalize: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the embedding model wrapper.

        Args:
            model_name: HuggingFace model identifier or local path.
            device: Computing device ('cpu', 'cuda').
            normalize: Whether to normalize output embeddings to unit L2 norm.
            cache_dir: Directory to persist precomputed embedding indices.
        """
        self.model_name = model_name
        self.device = device
        self.normalize = normalize

        if cache_dir is None:
            base_dir = Path(__file__).resolve().parent.parent.parent.parent
            self.cache_dir = base_dir / "data" / "indexes"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None  # Lazy loaded

    def _get_model(self):
        """Lazy loader: loads the sentence-transformers model into memory only when needed."""
        if self._model is None:
            logger.info(f"Loading embedding model '{self.model_name}' on {self.device}...")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Embedding model '{self.model_name}' loaded successfully.")
        return self._model

    def encode_documents(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode a batch of document texts into dense vectors.

        Args:
            texts: List of text strings to encode.
            batch_size: Number of texts to process in each forward pass.

        Returns:
            np.ndarray: Matrix of shape (N, D) where N is len(texts) and D is embedding dimension.
        """
        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Ensure float32 numpy array
        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Fallback manual normalization if not normalized by model
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            embeddings = embeddings / norms

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single search query into a dense vector.

        Args:
            query: The user query string.

        Returns:
            np.ndarray: 1D normalized vector of shape (D,).
        """
        if not query or not query.strip():
            # Return empty/zero vector
            dummy = np.zeros(768, dtype=np.float32)
            return dummy

        # BGE models benefit from instruction prefix for retrieval if configured,
        # but to remain fully general we encode the query string directly.
        model = self._get_model()
        embedding = model.encode(
            query,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        if self.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding

    @staticmethod
    def similarity(query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a query embedding and multiple document embeddings.

        Since embeddings are normalized to unit length (||q|| = 1, ||d_i|| = 1),
        the cosine similarity reduces directly to the dot product:
            cos(q, d_i) = q · d_i

        Args:
            query_embedding: 1D vector (D,) or 2D vector (1, D).
            doc_embeddings: 2D matrix (N, D) of document vectors.

        Returns:
            np.ndarray: 1D array of length N containing cosine similarity scores in [-1.0, 1.0].
        """
        if doc_embeddings.size == 0:
            return np.empty((0,), dtype=np.float32)

        q = np.asarray(query_embedding, dtype=np.float32).flatten()
        docs = np.asarray(doc_embeddings, dtype=np.float32)

        if docs.ndim == 1:
            docs = docs.reshape(1, -1)

        # If vectors are normalized, dot product equals cosine similarity
        scores = np.dot(docs, q)
        return scores

    @staticmethod
    def _compute_cache_key(chunks: List[DocumentChunk], model_name: str) -> str:
        """Generate a SHA-256 hash identifying the chunks and model."""
        hasher = hashlib.sha256()
        hasher.update(model_name.encode("utf-8"))
        for c in chunks:
            hasher.update(c.chunk_id.encode("utf-8"))
            hasher.update(c.text.encode("utf-8"))
        return hasher.hexdigest()[:16]

    def get_or_compute_chunk_embeddings(
        self,
        chunks: List[DocumentChunk],
        cache_name: Optional[str] = None,
    ) -> np.ndarray:
        """Retrieve precomputed chunk embeddings from disk cache or compute and cache them.

        Args:
            chunks: List of DocumentChunk objects to encode.
            cache_name: Optional custom filename stem for the cache.

        Returns:
            np.ndarray: Matrix of embeddings (N, D).
        """
        if not chunks:
            return np.empty((0, 768), dtype=np.float32)

        key = self._compute_cache_key(chunks, self.model_name)
        stem = cache_name or f"emb_{key}"
        cache_file = self.cache_dir / f"{stem}.npz"

        # Check if cache exists
        if cache_file.exists():
            try:
                data = np.load(str(cache_file), allow_pickle=True)
                cached_ids = list(data["chunk_ids"])
                current_ids = [c.chunk_id for c in chunks]
                if cached_ids == current_ids:
                    logger.info(f"Loaded {len(chunks)} cached embeddings from {cache_file.name}")
                    return data["embeddings"]
            except Exception as e:
                logger.warning(f"Failed to read cache {cache_file}: {e}. Recomputing.")

        # Compute embeddings
        logger.info(f"Encoding {len(chunks)} document chunks...")
        texts = [c.text for c in chunks]
        embeddings = self.encode_documents(texts)

        # Save to cache
        try:
            chunk_ids = np.array([c.chunk_id for c in chunks])
            np.savez_compressed(str(cache_file), embeddings=embeddings, chunk_ids=chunk_ids)
            logger.info(f"Saved {len(chunks)} embeddings to cache {cache_file.name}")
        except Exception as e:
            logger.warning(f"Could not persist embeddings cache to {cache_file}: {e}")

        return embeddings


def get_embedding_model(config: Optional[EmbeddingConfig] = None) -> EmbeddingModel:
    """Factory helper to obtain an EmbeddingModel instance configured via config.yaml.

    Args:
        config: Optional EmbeddingConfig. If None, loaded from central config.

    Returns:
        EmbeddingModel: Configured embedding model.
    """
    if config is None:
        app_cfg = load_config()
        config = app_cfg.embedding

    return EmbeddingModel(
        model_name=config.model_name,
        device=config.device,
        normalize=config.normalize,
    )
