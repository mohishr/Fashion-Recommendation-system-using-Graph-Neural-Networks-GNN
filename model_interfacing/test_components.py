import os
import sys

# Test importing the extractor
try:
    print("Testing Feature Extractor...")
    from feature_extractor import FeatureExtractor
    extractor = FeatureExtractor()
    print("Feature Extractor initialized successfully!")
except Exception as e:
    print(f"Feature Extractor Error: {e}")

# Test importing the predictor
try:
    print("\nTesting Outfit Predictor (Model Initializer)...")
    from inference import OutfitPredictor
    predictor = OutfitPredictor()
    print("Outfit Predictor initialized successfully!")
    
    # Try a dummy prediction
    import numpy as np
    dummy_imgs = np.zeros((1, 10, 2048), dtype=np.float32)
    dummy_txts = np.zeros((1, 10, 2757), dtype=np.float32)
    dummy_grph = np.ones((1, 10, 10), dtype=np.float32)
    
    print("\nRunning test forward pass...")
    score = predictor.predict(dummy_imgs, dummy_txts, dummy_grph)
    print(f"Test Score Result: {score}")
    
except Exception as e:
    print(f"Outfit Predictor Error: {e}")

# Test importing the recommender
try:
    print("\nTesting Recommender Engine...")
    from recommendation_engine import RecommenderEngine
    if 'predictor' in locals():
        recommender = RecommenderEngine(predictor)
        print("Recommender Engine initialized successfully!")
except Exception as e:
    print(f"Recommender Engine Error: {e}")
