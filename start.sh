#!/bin/bash
cd /home/manodhithyaa-c-s/Desktop/ULTRON/agents
nohup /home/manodhithyaa-c-s/Desktop/ULTRON/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 >> /tmp/ultron.log 2>&1 &
echo $!
