from tensorflow import keras

if __name__ == "__main__":
    _, (images, labels) = keras.datasets.cifar10.load_data()
    images = images.reshape(10000, 32, 32, 3).astype('float32') / 255
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')