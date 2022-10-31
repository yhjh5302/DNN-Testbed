# DNN-Testbed

This repository includes the source code used in the project  
"Partition Placement and Resource Allocation for Multiple DNN-based Applications in Heterogeneous IoT Environments".

## About the project

This project includes an entire testbed environment, such as
*  DNN model partitioning implementation for real-world testbed
*  DNN partition containerization
*  DNN container deployment and resource allocation using kubernetes

## Prerequisites
*  Linux OS
*  kubelet, kubeadm, kubectl, kubernetes-cni
*  python>=3.6, pytorch, numpy
*  tensorflow==2.5.0
*  *(optional)* Nvidia GPU, nvidia-container-runtime, nvidia-docker2
*  *(optional)* Plenty of free memory *(for each nvidia-container, 1GB memory is required for cuda kernel)*
*  *(optional)* Nvidia-MPS

## How to run
#### 1. Build container
>  cd tensorflow/AlexNet/layer (for example)  
>  docker build -t {dockerhub_ID}/{container_name} .  
>  docker push {dockerhub_ID}/{container_name}  

#### 2. Run container
>  docker pull {dockerhub_ID}/{container_name}  
>  sudo docker run -i --runtime=nvidia {dockerhub_ID}/{container_name}

or

>  kubectl apply -f pipeline-deploy.yaml  

If you want to use .yaml file, you have to change the container name  
