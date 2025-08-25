FROM pytorch/pytorch:latest
WORKDIR /workspace
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt
EXPOSE 8888
CMD ["bash","-lc","jupyter lab --ip=0.0.0.0 --no-browser --allow-root --notebook-dir=/workspace"]
