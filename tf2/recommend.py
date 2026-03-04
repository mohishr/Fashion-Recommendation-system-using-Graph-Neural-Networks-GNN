import numpy as np

try:
    import pymongo
except Exception:
    pymongo = None

# Configurations
MONGO_URI = 'mongodb://localhost:27021/'
DB_NAME = 'fashion_recommendation_db'
COLLECTION_NAME = 'Clothing_Items'


class RecommenderEngine:
    def __init__(self, outfit_api_instance):
        """
        outfit_api_instance = OutfitCompatibilityAPI()
        """
        self.api = outfit_api_instance
        self.collection = None
        self.max_items = 10

        if pymongo is None:
            print('WARNING: pymongo is not installed. /api/recommend_item will be disabled.')
            return

        try:
            self.client = pymongo.MongoClient(MONGO_URI)
            self.db = self.client[DB_NAME]
            self.collection = self.db[COLLECTION_NAME]
            print('RecommenderEngine connected to MongoDB.')
        except Exception as e:
            print('Failed to connect to MongoDB: {}'.format(e))

    # ==========================================================
    # CORE FUNCTION
    # ==========================================================
    def get_recommendations_for_outfit(self, partial_outfit, target_category, top_n=10):
        """
        partial_outfit:
            [
                {
                    "image_embedding": [...2048...],
                },
                ...
            ]

        target_category:
            "shoes", "tops", etc.

        Returns:
            Top N ranked compatible items
        """

        if self.collection is None:
            return []

        print(f'Fetching candidates for category: {target_category}...')

        cursor = self.collection.find(
            {'category': target_category},
            {
                '_id': 1,
                'category': 1,
                'description': 1,
                'image_embedding': 1,
                'image_blob': 1
            },
        )

        candidates = list(cursor)

        if not candidates:
            return []

        print(f'Found {len(candidates)} candidates. Evaluating scores...')

        scored_candidates = []

        for candidate in candidates:

            if not candidate.get('image_embedding'):
                continue

            try:
                # ------------------------------------------------
                # Build outfit: partial + candidate
                # ------------------------------------------------
                images = []

                # Existing outfit items
                for item in partial_outfit:
                    images.append(
                        np.array(item['image_embedding'], dtype=np.float32)
                    )

                # Add candidate
                images.append(
                    np.array(candidate['image_embedding'], dtype=np.float32)
                )

                num_items = len(images)

                # ------------------------------------------------
                # Pad to max_items
                # ------------------------------------------------
                while len(images) < self.max_items:
                    images.append(np.zeros(2048, dtype=np.float32))

                images_input = np.array([images])  # shape (1,10,2048)

                # ------------------------------------------------
                # Build graph (fully connected for real items)
                # ------------------------------------------------
                graph = np.zeros((self.max_items, self.max_items), dtype=np.float32)
                graph[:num_items, :num_items] = 1.0
                graph_input = np.array([graph])  # shape (1,10,10)

                # ------------------------------------------------
                # Predict using new TF2 model
                # ------------------------------------------------
                score_tensor = self.api.model(images_input, graph_input)
                score = float(score_tensor.numpy()[0][0])

                scored_candidates.append({
                    'item_id': candidate['_id'],
                    'category': candidate['category'],
                    'description': candidate['description'],
                    'score': score,
                })

            except Exception as e:
                print(f"Skipping candidate due to error: {e}")
                continue

        # ------------------------------------------------
        # Sort by compatibility score (descending)
        # ------------------------------------------------
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)

        top_candidates = scored_candidates[:top_n]

        # ------------------------------------------------
        # Attach image blobs (same logic as before)
        # ------------------------------------------------
        import base64

        top_ids = [c['item_id'] for c in top_candidates]

        blob_cursor = self.collection.find(
            {'_id': {'$in': top_ids}},
            {'_id': 1, 'image_blob': 1}
        )

        blob_map = {doc['_id']: doc.get('image_blob') for doc in blob_cursor}

        for candidate in top_candidates:
            blob = blob_map.get(candidate['item_id'])

            if blob:
                candidate['image_base64'] = base64.b64encode(blob).decode('utf-8')
            else:
                candidate['image_base64'] = None

        return top_candidates