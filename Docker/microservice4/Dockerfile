FROM python:3.9-slim

WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0

COPY . /app

RUN pip install -r requirements.txt