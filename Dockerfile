FROM python:3.9-slim

WORKDIR /app

# Install te sytem dependencies

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copy local folders into the container
COPY app/ ./app/
COPY scripts/ ./scripts/

EXPOSE 8000

#Run the app
CMD [ "uvicorn", "app.main:app", "--host","0.0.0.0", "--port", 8000]