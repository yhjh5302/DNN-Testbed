FROM nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.5-py3
ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code
COPY . /code/
RUN python3 data_downloader.py
RUN rm data_downloader.py
RUN apt install nano
