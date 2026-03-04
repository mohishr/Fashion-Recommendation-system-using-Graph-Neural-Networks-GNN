# TensorFlow 1.x base image (compatible with Session/placeholder style code in this repo)
FROM tensorflow/tensorflow:1.15.5-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NGNN_DISABLE_IMAGENET=1

WORKDIR /app

# Install runtime dependencies for Flask app + feature/recommendation helpers
COPY model_interfacing/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir pymongo pillow

# Copy project
COPY . /app

WORKDIR /app/model_interfacing
EXPOSE 5000

# Run the interface
CMD ["python", "app.py"]
