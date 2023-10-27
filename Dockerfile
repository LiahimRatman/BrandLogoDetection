FROM python:3.8

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
COPY run.py /run.py

ENV PYTHONPATH "${PYTHONPATH}:/"

CMD uvicorn run:app --host=0.0.0.0 --port=8021
