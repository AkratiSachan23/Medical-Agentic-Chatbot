from functools import lru_cache
from pinecone import Pinecone, ServerlessSpec
from app.config import settings


@lru_cache()
def pinecone_client():
    if not settings.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY must be set in the environment variables.")
    return Pinecone(api_key=settings.pinecone_api_key)


@lru_cache()
def embedding_model():
    """Load the free local model once, on its first use."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Local embeddings require sentence-transformers. "
            "Run: python -m pip install -r requirements.txt"
        ) from exc
    return SentenceTransformer(settings.embedding_model)


def ensure_index():
    pc = pinecone_client()
    if not pc.has_index(settings.pinecone_index):
        pc.create_index(
            name=settings.pinecone_index,
            dimension=settings.embedding_dimension,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1'),
        )
    else:
        index_info = pc.describe_index(settings.pinecone_index)
        if index_info.dimension != settings.embedding_dimension:
            raise ValueError(
                f"Pinecone index '{settings.pinecone_index}' has dimension "
                f"{index_info.dimension}, but local embeddings need "
                f"{settings.embedding_dimension}. Set PINECONE_INDEX to a new "
                "index name, such as medical-agent-local, then rerun ingestion."
            )
    return pc.Index(settings.pinecone_index)


def embed(texts):
    # normalize_embeddings makes dot product equivalent to cosine similarity.
    return embedding_model().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    ).tolist()



def search_kb(query, top_k=5):
    index = ensure_index()
    vector = embed([query])[0]
    result = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [{'page': m.metadata.get('page'), 'text': m.metadata.get('text',''), 'score': round(m.score,3), 'source': 'Gale Encyclopedia (2002)'} for m in result.matches if m.score >= 0.30]
