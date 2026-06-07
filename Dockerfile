# Container image for deploying the Sudoku web app (e.g. on Hugging Face Spaces).
FROM python:3.11-slim

# System libraries:
#  - fonts-* : real digit fonts so train.py can build the printed-digit dataset
#  - libgl1 / libglib2.0-0 : required by OpenCV (opencv-python)
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-liberation fonts-dejavu-core fonts-freefont-ttf \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

# Use a committed model if one is present (so the deployed model matches what
# you trained/validated locally); otherwise train one at build time as a
# fallback using the fonts installed above.
RUN test -f models/digit_model.keras -o -f models/digit_model.h5 \
    || python train.py --epochs 10

# Hugging Face Spaces expects the app on port 7860.
EXPOSE 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "-w", "1", "--threads", "4", \
     "--timeout", "120", "app:app"]
