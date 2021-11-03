FROM tensorflow/tensorflow:latest-gpu
ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code
COPY . /code/
RUN apt update
RUN apt-get install nano
RUN apt-get install sshpass -y
