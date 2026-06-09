FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY main.py .

ENV MODEL_SIZE=large-v3
ENV PORT=8082

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8082"]
