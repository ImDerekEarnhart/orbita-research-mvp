FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir .

ENV ORBITA_MVP_DB=/data/orbita_mvp.db
ENV ORBITA_MVP_WORKSPACE=/data/orbita_workspace

EXPOSE 8010

CMD ["orbita-research-api"]
