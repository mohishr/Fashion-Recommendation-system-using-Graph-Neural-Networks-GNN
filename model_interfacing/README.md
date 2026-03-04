# Model Interfacing App

This Flask app provides a working interface to score outfit compatibility using the NGNN checkpoint.

## 1) Prepare checkpoint files

Place TensorFlow checkpoint files in `model_interfacing/model_weights/`.

Accepted examples:
- `cm_ggnn.ckpt.index` + `cm_ggnn.ckpt.meta` + `cm_ggnn.ckpt.data-*`
- `model.ckpt-34865.index` + `model.ckpt-34865.meta` + `model.ckpt-34865.data-*`

You can also set one of these environment variables:
- `NGNN_CKPT_PATH` (full checkpoint prefix path)
- `NGNN_WEIGHTS_DIR` (directory to search)
- `NGNN_CHECKPOINT_NAME` (checkpoint prefix file name under weights dir)

## 2) Install dependencies

```bash
pip install -r model_interfacing/requirements.txt
```

Install TensorFlow 1.x compatible with your environment.

## 3) Run

```bash
cd model_interfacing
python app.py
```

Open: `http://localhost:5000/`

## 4) API usage

Endpoint: `POST /api/predict`

```json
{
  "items": [
    {"text": "black t-shirt", "image_path": "/abs/path/item1.jpg"},
    {"text": "blue denim jeans", "image_path": "/abs/path/item2.jpg"}
  ]
}
```

Additional status endpoint: `GET /api/model_status`


## 5) Smoke test with dummy images

From `model_interfacing/`:

```bash
NGNN_DISABLE_IMAGENET=1 python smoke_test_api.py
```

This posts to `/api/predict` with dummy/non-existent image paths. The extractor gracefully falls back to zero image features, so you can still verify end-to-end API scoring and model loading behavior.


## 6) Preflight check (recommended in restricted environments)

```bash
cd model_interfacing
python verify_environment.py
```

This checks:
- required Python modules (`flask`, `numpy`, `tensorflow`)
- optional module (`pymongo`)
- available checkpoints under `model_weights/`
- exact command to launch the server

If this reports missing modules, install them in your terminal before running `app.py`.


## 7) Run in a TensorFlow base-image container

A ready Dockerfile is provided at `model_interfacing/Dockerfile` and uses:
- `tensorflow/tensorflow:1.15.5-py3` (matches TF1-style `Session`/`placeholder` code)

Build image (from repo root):

```bash
docker build -f model_interfacing/Dockerfile -t ngnn-interface:tf1 .
```

Run container:

```bash
docker run --rm -p 5000:5000   -e NGNN_DISABLE_IMAGENET=1   -e NGNN_WEIGHTS_DIR=/app/model_interfacing/model_weights   -v "$(pwd)/model_interfacing/model_weights:/app/model_interfacing/model_weights"   ngnn-interface:tf1
```

Then open `http://localhost:5000`.

Optional: point directly to a checkpoint prefix:

```bash
docker run --rm -p 5000:5000   -e NGNN_CKPT_PATH=/app/model_interfacing/model_weights/cm_ggnn.ckpt   -v "$(pwd)/model_interfacing/model_weights:/app/model_interfacing/model_weights"   ngnn-interface:tf1
```
