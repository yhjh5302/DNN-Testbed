sudo docker rmi yhjh5302/alexnet-layer1-nano && \
sudo docker rmi yhjh5302/alexnet-layer2-nano && \
sudo docker rmi yhjh5302/alexnet-layer3-nano && \
sudo docker rmi yhjh5302/googlenet-layer1-nano && \
sudo docker rmi yhjh5302/googlenet-layer2-nano && \
sudo docker rmi yhjh5302/googlenet-layer3-nano && \
sudo docker rmi yhjh5302/mobilenet-layer1-nano && \
sudo docker rmi yhjh5302/mobilenet-layer2-nano && \
sudo docker rmi yhjh5302/mobilenet-layer3-nano && \
sudo docker rmi yhjh5302/vggnet-layer1-nano && \
sudo docker rmi yhjh5302/vggnet-layer2-nano && \
sudo docker rmi yhjh5302/vggnet-layer3-nano && \
sudo docker rmi yhjh5302/vggfnet-layer1-nano && \
sudo docker rmi yhjh5302/vggfnet-layer2-nano && \
cd tensorflow-nano/AlexNet/layer1 && sudo docker build -t yhjh5302/alexnet-layer1-nano . && \
cd ../../AlexNet/layer2 && sudo docker build -t yhjh5302/alexnet-layer2-nano . && \
cd ../../AlexNet/layer3 && sudo docker build -t yhjh5302/alexnet-layer3-nano . && \
cd ../../GoogLeNet/layer1 && sudo docker build -t yhjh5302/googlenet-layer1-nano . && \
cd ../../GoogLeNet/layer2 && sudo docker build -t yhjh5302/googlenet-layer2-nano . && \
cd ../../GoogLeNet/layer3 && sudo docker build -t yhjh5302/googlenet-layer3-nano . && \
cd ../../MobileNet/layer1 && sudo docker build -t yhjh5302/mobilenet-layer1-nano . && \
cd ../../MobileNet/layer2 && sudo docker build -t yhjh5302/mobilenet-layer2-nano . && \
cd ../../MobileNet/layer3 && sudo docker build -t yhjh5302/mobilenet-layer3-nano . && \
cd ../../VGGNet/layer1 && sudo docker build -t yhjh5302/vggnet-layer1-nano . && \
cd ../../VGGNet/layer2 && sudo docker build -t yhjh5302/vggnet-layer2-nano . && \
cd ../../VGGNet/layer3 && sudo docker build -t yhjh5302/vggnet-layer3-nano . && \
cd ../../VGGFNet/layer1 && sudo docker build -t yhjh5302/vggfnet-layer1-nano . && \
cd ../../VGGFNet/layer2 && sudo docker build -t yhjh5302/vggfnet-layer2-nano . && \
sudo docker push yhjh5302/alexnet-layer1-nano && \
sudo docker push yhjh5302/alexnet-layer2-nano && \
sudo docker push yhjh5302/alexnet-layer3-nano && \
sudo docker push yhjh5302/googlenet-layer1-nano && \
sudo docker push yhjh5302/googlenet-layer2-nano && \
sudo docker push yhjh5302/googlenet-layer3-nano && \
sudo docker push yhjh5302/mobilenet-layer1-nano && \
sudo docker push yhjh5302/mobilenet-layer2-nano && \
sudo docker push yhjh5302/mobilenet-layer3-nano && \
sudo docker push yhjh5302/vggnet-layer1-nano && \
sudo docker push yhjh5302/vggnet-layer2-nano && \
sudo docker push yhjh5302/vggnet-layer3-nano && \
sudo docker push yhjh5302/vggfnet-layer1-nano && \
sudo docker push yhjh5302/vggfnet-layer2-nano