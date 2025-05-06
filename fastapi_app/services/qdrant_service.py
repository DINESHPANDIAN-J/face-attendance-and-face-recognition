# fastapi_app/services/qdrant_service.py

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import numpy as np
import os

client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=6333)
COLLECTION_NAME = "faces"

def save_user_to_qdrant(user_id: str, name: str, embedding: np.ndarray):
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE)
    )

    point = PointStruct(
        id=user_id,
        vector=embedding.tolist(),
        payload={"name": name, "user_id": user_id}
    )
    client.upsert(collection_name=COLLECTION_NAME, points=[point])
