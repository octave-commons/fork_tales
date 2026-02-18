ARG TF_IMAGE=tensorflow/tensorflow:2.16.1
FROM ${TF_IMAGE}

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /workspace

CMD ["python", "train.py", "--config", "/workspace/config.yaml"]
