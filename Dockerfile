FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV VLLM_URL=http://vllm:8000/v1 WHISPER_DEVICE=cpu
EXPOSE 8000
CMD ["uvicorn", "voxflow.server:app", "--host", "0.0.0.0", "--port", "8000"]
