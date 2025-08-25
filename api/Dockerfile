FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime
WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

EXPOSE 8000
CMD ["uvicorn","api.server:app","--host","0.0.0.0","--port","8000"]
