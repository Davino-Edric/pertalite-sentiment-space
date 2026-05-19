FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY model.pkl .

RUN useradd -m -u 1000 appuser

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

CMD ["python", "app.py"]