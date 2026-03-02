import pymongo
import numpy as np

# Configurations (Ensure these match your seed_db.py settings)
MONGO_URI = "mongodb://localhost:27021/"
DB_NAME = "fashion_recommendation_db"
COLLECTION_NAME = "Clothing_Items"

class RecommenderEngine:
    def __init__(self, predictor_instance):
        """
        Takes an instance of OutfitPredictor so we can run models.
        Connects to MongoDB to fetch candidate items.
        """
        self.predictor = predictor_instance
        try:
            self.client = pymongo.MongoClient(MONGO_URI)
            self.db = self.client[DB_NAME]
            self.collection = self.db[COLLECTION_NAME]
            print("RecommenderEngine connected to MongoDB.")
        except Exception as e:
            print(f"Failed to connect to MongoDB in RecommenderEngine: {e}")

    def get_recommendations_for_outfit(self, partial_outfit, target_category, top_n=10):
        """
        Given a partial outfit, finds the best scoring items from the database 
        in the target_category to complete the look.
        
        Args:
            partial_outfit (list of dicts): The outfit we have so far
                e.g., [{"image_embedding": [...], "text_embedding": [...]}]
            target_category (str): the literal category name to search DB for
                e.g., "Jeans", "Shoes", "Bags"
            top_n (int): Max number of recommendations to return
            
        Returns:
            list of dicts containing the recommended item details and their score.
        """
        print(f"Fetching candidates for category: {target_category}...")
        
        # 1. Fetch Candidate Items from DB
        # To avoid pulling thousands of blobs into memory, we only pull the embeddings and basic info
        cursor = self.collection.find(
            {"category": target_category}, 
            {"_id": 1, "category": 1, "description": 1, "image_embedding": 1, "text_embedding": 1}
        )
        candidates = list(cursor)
        
        if not candidates:
            return []
            
        print(f"Found {len(candidates)} candidates. Evaluating scores...")
        
        # 2. Score each candidate
        scored_candidates = []
        
        for index, candidate in enumerate(candidates):
            # Skip candidates missing embeddings
            if not candidate.get("image_embedding"):
                continue
                
            # Build text embedding (fallback to zeros if None like in seed_db)
            img_embed = np.array(candidate["image_embedding"], dtype=np.float32)
            if candidate.get("text_embedding"):
                txt_embed = np.array(candidate["text_embedding"], dtype=np.float32)
            else:
                txt_embed = np.zeros(2757, dtype=np.float32)
            
            # Construct a test outfit by appending the candidate to the partial outfit
            # Format must match what predictor.predict expects
            
            # The predictor logic extracts features from dicts via: 
            # item.get('image_path') 
            # Since we ALREADY have the embeddings here straight from DB, 
            # we need to slightly bridge the gap.
            
            # Let's format the raw arrays
            num_items = len(partial_outfit) + 1
            images = []
            texts = []
            
            # Add existing outfit items
            for item in partial_outfit:
                images.append(np.array(item["image_embedding"], dtype=np.float32))
                texts.append(np.array(item["text_embedding"], dtype=np.float32))
                
            # Add candidate
            images.append(img_embed)
            texts.append(txt_embed)
            
            # Pad
            MAX_ITEMS = 10
            while len(images) < MAX_ITEMS:
                images.append(np.zeros(2048, dtype=np.float32))
                texts.append(np.zeros(2757, dtype=np.float32))
                
            # Build Graph
            graph = np.zeros((MAX_ITEMS, MAX_ITEMS), dtype=np.float32)
            for i in range(num_items):
                for j in range(num_items):
                    graph[i, j] = 1.0
                    
            # Predict
            images_input = np.array([images])
            texts_input = np.array([texts])
            graph_input = np.array([graph])
            
            try:
                score = self.predictor.predict(images_input, texts_input, graph_input)
                
                # Format result
                result = {
                    "item_id": candidate["_id"],
                    "category": candidate["category"],
                    "description": candidate["description"],
                    "score": float(score)
                }
                scored_candidates.append(result)
            except Exception as e:
                pass # skip if inference fails
                
        # 3. Sort best to worst
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = scored_candidates[:top_n]
        
        # 4. Fetch the image blob only for the top N
        import base64
        top_ids = [c["item_id"] for c in top_candidates]
        blob_cursor = self.collection.find(
            {"_id": {"$in": top_ids}}, 
            {"_id": 1, "image_blob": 1}
        )
        blob_map = {doc["_id"]: doc.get("image_blob") for doc in blob_cursor}

        for candidate in top_candidates:
            blob = blob_map.get(candidate["item_id"])
            if blob:
                candidate["image_base64"] = base64.b64encode(blob).decode('utf-8')
            else:
                candidate["image_base64"] = None
        
        return top_candidates
