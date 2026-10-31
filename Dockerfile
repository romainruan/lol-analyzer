FROM python:3.12-slim

# les messages Python s'affichent tout de suite dans `docker compose logs`
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY tests ./tests

# sans argument, le conteneur affiche l'aide du débrief
CMD ["python", "-m", "src.debrief", "--help"]
