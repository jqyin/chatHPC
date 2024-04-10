#!/bin/bash
docker build -t registry.apps.marble.ccs.ornl.gov/stf218-app/chathpc:latest . &&
docker push registry.apps.marble.ccs.ornl.gov/stf218-app/chathpc:latest &&
# oc --namespace stf218-app delete statefulset chathpc
# Scale down so pod gets recreated an uses new image
oc --namespace stf218-app scale statefulset -l app=chathpc --replicas=0
oc apply -f chathpc.yaml
# oc --namespace stf218-app rsh chathpc-0 bash;
