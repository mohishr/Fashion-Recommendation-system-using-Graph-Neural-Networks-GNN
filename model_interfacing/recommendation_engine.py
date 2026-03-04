import numpy as np

try:
    import pymongo
except Exception:
    pymongo = None

# Configurations (Ensure these match your seed_db.py settings)
MONGO_URI = 'mongodb://localhost:27021/'
DB_NAME = 'fashion_recommendation_db'
COLLECTION_NAME = 'Clothing_Items'


class RecommenderEngine:
    def __init__(self, predictor_instance):
        self.predictor = predictor_instance
        self.collection = None

        if pymongo is None:
            print('WARNING: pymongo is not installed. /api/recommend_item will be disabled.')
            return

        try:
            self.client = pymongo.MongoClient(MONGO_URI)
            self.db = self.client[DB_NAME]
            self.collection = self.db[COLLECTION_NAME]
            print('RecommenderEngine connected to MongoDB.')
        except Exception as e:
            print('Failed to connect to MongoDB in RecommenderEngine: {}'.format(e))

    def get_recommendations_for_outfit(self, partial_outfit, target_category, top_n=10):
        if self.collection is None:
            return []

        print('Fetching candidates for category: {}...'.format(target_category))

        cursor = self.collection.find(
            {'category': target_category},
            {'_id': 1, 'category': 1, 'description': 1, 'image_embedding': 1, 'text_embedding': 1},
        )
        candidates = list(cursor)

        if not candidates:
            return []

        print('Found {} candidates. Evaluating scores...'.format(len(candidates)))

        scored_candidates = []

        for candidate in candidates:
            if not candidate.get('image_embedding'):
                continue

            img_embed = np.array(candidate['image_embedding'], dtype=np.float32)
            if candidate.get('text_embedding'):
                txt_embed = np.array(candidate['text_embedding'], dtype=np.float32)
            else:
                txt_embed = np.zeros(2757, dtype=np.float32)

            num_items = len(partial_outfit) + 1
            images = []
            texts = []

            for item in partial_outfit:
                images.append(np.array(item['image_embedding'], dtype=np.float32))
                texts.append(np.array(item['text_embedding'], dtype=np.float32))

            images.append(img_embed)
            texts.append(txt_embed)

            max_items = 10
            while len(images) < max_items:
                images.append(np.zeros(2048, dtype=np.float32))
                texts.append(np.zeros(2757, dtype=np.float32))

            graph = np.zeros((max_items, max_items), dtype=np.float32)
            for i in range(num_items):
                for j in range(num_items):
                    graph[i, j] = 1.0

            images_input = np.array([images])
            texts_input = np.array([texts])
            graph_input = np.array([graph])

            try:
                score = self.predictor.predict(images_input, texts_input, graph_input)
                result = {
                    'item_id': candidate['_id'],
                    'category': candidate['category'],
                    'description': candidate['description'],
                    'score': float(score),
                }
                scored_candidates.append(result)
            except Exception:
                pass

        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        top_candidates = scored_candidates[:top_n]

        import base64

        top_ids = [c['item_id'] for c in top_candidates]
        blob_cursor = self.collection.find({'_id': {'$in': top_ids}}, {'_id': 1, 'image_blob': 1})
        blob_map = {doc['_id']: doc.get('image_blob') for doc in blob_cursor}

        for candidate in top_candidates:
            blob = blob_map.get(candidate['item_id'])
            if blob:
                candidate['image_base64'] = base64.b64encode(blob).decode('utf-8')
            else:
                candidate['image_base64'] = None

        return top_candidates
