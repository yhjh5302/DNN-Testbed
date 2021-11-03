FROM tensorflow/tensorflow:latest-gpu
ENV PYTHONUNBUFFERED 1
RUN apt update; apt install sshpass -y; apt install nano -y;
RUN mkdir /code
WORKDIR /code
COPY . /code
