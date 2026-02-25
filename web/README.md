# Web Interface

This folder contains a lightweight Flask app to expose three recommendation flows mapped to the repository's NGNN tasks:

1. **Compatibility score** (`/compatibility`)  
2. **Fill-in-the-blank** item suggestion (`/fill-in-blank`)  
3. **Multimodal ranking** (`/multimodal`) combining text and category preferences.

## Run

```bash
pip install -r web/requirements.txt
python web/app.py
```

Open: `http://localhost:5000`

## Notes

- The original training code depends on TensorFlow 1.x and heavy feature files. This web UI uses available JSON metadata and sample images from `data/polyvore-images_smallsample` for quick exploration.
- When no fully matched metadata+image outfits exist, the app creates fallback sample outfits from local images so the UI remains usable.
