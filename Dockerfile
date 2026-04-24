FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY data.csv .
COPY start.sh .

RUN mkdir -p artifacts && chmod +x start.sh

EXPOSE 8000

ENTRYPOINT ["/bin/sh", "start.sh"]
