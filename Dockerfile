FROM python:3.12-slim

# les messages Python s'affichent tout de suite dans `docker compose logs`
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY src ./src
COPY tests ./tests

# sans argument, le conteneur affiche l'aide du débrief
CMD ["python", "-m", "src.debrief", "--help"]
