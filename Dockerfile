FROM python:3.11-slim
WORKDIR /app
ARG WITH_MINILM=0
COPY requirements.txt requirements-minilm.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && if [ "$WITH_MINILM" = "1" ]; then pip install --no-cache-dir -r requirements-minilm.txt; fi
COPY . .
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
