import numpy as np
import pymongo
import base64

MONGO_URI = 'mongodb://localhost:27021/'
DB_NAME = 'fashion_recommendation_db'
COLLECTION_NAME = 'Clothing_Items'

class RecommenderEngine:
    def __init__(self):
        self.client = pymongo.MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
        self.predictor = None
        self.extractor = None
        print("RecommenderEngine connected to MongoDB.")

    @staticmethod
    def cosine_similarity(a, b):
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_recommendations_for_outfit(self, partial_outfit, target_category, top_n=5):
        if not partial_outfit:
            return []

        candidates = list(self.collection.find(
            {"category": target_category},
            {"_id": 1, "category": 1, "description": 1, "image_embedding": 1, "image_blob": 1}
        ))

        if not candidates:
            return []

        scored = []
        for candidate in candidates:
            if not candidate.get("image_embedding"):
                continue

            candidate_embedding = np.array(candidate["image_embedding"], dtype=np.float32)
            outfit_embeddings = [np.array(it["image_embedding"], dtype=np.float32) for it in partial_outfit]
            outfit_embeddings.append(candidate_embedding)
            try:
                raw_score = self.predictor.predict_from_embeddings(outfit_embeddings)
                normalized_score = max(0, min(100, raw_score * 10))
                scored.append({
                    "item_id": str(candidate["_id"]),
                    "category": candidate["category"],
                    "description": candidate.get("description", ""),
                    "score": float(normalized_score),
                    "image_base64": base64.b64encode(candidate["image_blob"]).decode("utf-8") if candidate.get("image_blob") else None
                })
            except Exception:
                continue

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_n]