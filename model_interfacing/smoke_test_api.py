"""Local smoke test for model_interfacing Flask API.

Runs a request against /api/predict with dummy image paths.
Requires: numpy + tensorflow installed.
"""

import json

from app import app, predictor


def main():
    client = app.test_client()

    payload = {
        'items': [
            {'text': 'black t-shirt', 'image_path': '/tmp/nonexistent_item_1.jpg'},
            {'text': 'blue denim jeans', 'image_path': '/tmp/nonexistent_item_2.jpg'},
        ]
    }

    res = client.post('/api/predict', data=json.dumps(payload), content_type='application/json')
    print('status_code=', res.status_code)
    print('body=', res.get_json())
    print('model_loaded=', predictor.is_loaded)
    print('loaded_checkpoint=', predictor.loaded_checkpoint)


if __name__ == '__main__':
    main()
