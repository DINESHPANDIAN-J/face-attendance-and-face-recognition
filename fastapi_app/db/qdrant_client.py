from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(host="localhost", port=6333)

def setup_collection(collection_name: str, vector_dim: int):
    if not client.collection_exists(collection_name):
        client.recreate_collection(
            collection_name = collection_name,
            vectors_config = VectorParams(size=vector_dim, distance=Distance.COSINE)
        )
        
def insert_face_embedding(collection, person_id, embedding):
    client.upsert(
        collection_name=collection,
        points=[{
            ""id": person_id,
            "vector": embedding.tolist(),
            "payload": {
                "person_id": person_id
            }
        }]
    )
