FROM nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.5-py3
ENV PYTHONUNBUFFERED 1
RUN apt update; apt install sshpass -y; apt install nano -y;
RUN mkdir /code
WORKDIR /code
COPY . /code/
