FROM python:3.10-slim

WORKDIR /image_classifier
COPY . /image_classifier

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
ENV FLASK_APP=image_classifier.app

CMD ["flask", "run", "--host=0.0.0.0"]
