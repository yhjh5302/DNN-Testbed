# FROM tensorflow/tensorflow:latest-gpu
FROM nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.5-py3
ENV PYTHONUNBUFFERED 1
# RUN mkdir /code
WORKDIR /root
COPY . /root/
RUN apt update; apt install sshpass nano cmake unzip -y
# Run sh build_opencv.sh
RUN ./OpenCV-4-6-0.sh
