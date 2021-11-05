sudo docker rmi yhjh5302/alexnet-layer1 && \
sudo docker rmi yhjh5302/alexnet-layer2 && \
sudo docker rmi yhjh5302/alexnet-layer3 && \
sudo docker rmi yhjh5302/googlenet-layer1 && \
sudo docker rmi yhjh5302/googlenet-layer2 && \
sudo docker rmi yhjh5302/googlenet-layer3 && \
sudo docker rmi yhjh5302/mobilenet-layer1 && \
sudo docker rmi yhjh5302/mobilenet-layer2 && \
sudo docker rmi yhjh5302/mobilenet-layer3 && \
sudo docker rmi yhjh5302/vggnet-layer1 && \
sudo docker rmi yhjh5302/vggnet-layer2 && \
sudo docker rmi yhjh5302/vggnet-layer3 && \
sudo docker rmi yhjh5302/vggfnet-layer1 && \
sudo docker rmi yhjh5302/vggfnet-layer2 && \
sudo docker rmi yhjh5302/data-generator && \
sudo docker rmi yhjh5302/scheduler && \
cd tensorflow/AlexNet/layer1 && sudo docker build -t yhjh5302/alexnet-layer1 . && \
cd ../../AlexNet/layer2 && sudo docker build -t yhjh5302/alexnet-layer2 . && \
cd ../../AlexNet/layer3 && sudo docker build -t yhjh5302/alexnet-layer3 . && \
cd ../../GoogLeNet/layer1 && sudo docker build -t yhjh5302/googlenet-layer1 . && \
cd ../../GoogLeNet/layer2 && sudo docker build -t yhjh5302/googlenet-layer2 . && \
cd ../../GoogLeNet/layer3 && sudo docker build -t yhjh5302/googlenet-layer3 . && \
cd ../../MobileNet/layer1 && sudo docker build -t yhjh5302/mobilenet-layer1 . && \
cd ../../MobileNet/layer2 && sudo docker build -t yhjh5302/mobilenet-layer2 . && \
cd ../../MobileNet/layer3 && sudo docker build -t yhjh5302/mobilenet-layer3 . && \
cd ../../VGGNet/layer1 && sudo docker build -t yhjh5302/vggnet-layer1 . && \
cd ../../VGGNet/layer2 && sudo docker build -t yhjh5302/vggnet-layer2 . && \
cd ../../VGGNet/layer3 && sudo docker build -t yhjh5302/vggnet-layer3 . && \
cd ../../VGGFNet/layer1 && sudo docker build -t yhjh5302/vggfnet-layer1 . && \
cd ../../VGGFNet/layer2 && sudo docker build -t yhjh5302/vggfnet-layer2 . && \
cd ../../VGGFNet/layer2 && sudo docker build -t yhjh5302/vggfnet-layer2 . && \
cd ../../DataGenerator && sudo docker build -t yhjh5302/data-generator . && \
cd ../time-slicing-scheduler && sudo docker build -t yhjh5302/scheduler . && \
sudo docker push yhjh5302/alexnet-layer1 && \
sudo docker push yhjh5302/alexnet-layer2 && \
sudo docker push yhjh5302/alexnet-layer3 && \
sudo docker push yhjh5302/googlenet-layer1 && \
sudo docker push yhjh5302/googlenet-layer2 && \
sudo docker push yhjh5302/googlenet-layer3 && \
sudo docker push yhjh5302/mobilenet-layer1 && \
sudo docker push yhjh5302/mobilenet-layer2 && \
sudo docker push yhjh5302/mobilenet-layer3 && \
sudo docker push yhjh5302/vggnet-layer1 && \
sudo docker push yhjh5302/vggnet-layer2 && \
sudo docker push yhjh5302/vggnet-layer3 && \
sudo docker push yhjh5302/vggfnet-layer1 && \
sudo docker push yhjh5302/vggfnet-layer2 && \
sudo docker push yhjh5302/data-generator && \
sudo docker push yhjh5302/scheduler