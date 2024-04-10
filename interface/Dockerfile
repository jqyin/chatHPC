# docker build -t registry.apps.marble.ccs.ornl.gov/stf218-dev/chathpc:latest .
# docker push registry.apps.marble.ccs.ornl.gov/stf218-dev/chathpc:latest
# oc apply -f chathpc.yaml

# docker build -t registry.apps.marble.ccs.ornl.gov/stf218-dev/chathpc:latest .
# docker push registry.apps.marble.ccs.ornl.gov/stf218-dev/chathpc:latest
# oc apply -f chathpc.yaml

FROM python:3.9

RUN apt-get update \
  && apt-get -y install rsync git \
  && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt /app
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY FastChat /app/FastChat
RUN cd FastChat && \
    python -m pip install -e ".[model_worker,webui]" && \
    cd ..

COPY main.py /app/
COPY /chathpc /app/chathpc

CMD ["python", "main.py"]
