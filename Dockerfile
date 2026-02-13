FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data

ENV PYTHONUNBUFFERED=1
ENV MEMORY_EMB_DIM=384

EXPOSE 8000 19192

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
