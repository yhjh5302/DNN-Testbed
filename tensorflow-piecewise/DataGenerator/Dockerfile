FROM tensorflow/tensorflow:latest-gpu
ENV PYTHONUNBUFFERED 1
RUN apt update; apt install sshpass -y; apt install nano -y;
RUN mkdir /code
WORKDIR /code
COPY . /code/
RUN python3 data_downloader.py
RUN rm data_downloader.py
RUN apt install nano
