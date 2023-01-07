# rm -r /usr/include/opencv4/opencv2
make install
ldconfig

# cleaning (frees 300 MB)
make clean
apt-get update

echo "Congratulations!"
echo "You've successfully installed OpenCV 4.6.0 on your Jetson Nano"
