sleep 3;
python3 AlexNetLayer1.py --set_gpu='true' --prev_addr='' --prev_port=30001 --next_addr='10.96.0.202' --next_port=30002 --vram_limit=200 --debug=100 > AlexNetLayer1.txt;
python3 AlexNetLayer2.py --set_gpu='true' --prev_addr='' --prev_port=30002 --next_addr='10.96.0.203' --next_port=30003 --vram_limit=200 --debug=100 > AlexNetLayer2.txt;
python3 AlexNetLayer3.py --set_gpu='true' --prev_addr='' --prev_port=30003 --next_addr='10.96.0.200' --next_port=30000 --vram_limit=500 --debug=100 > AlexNetLayer3.txt;
python3 GoogLeNetLayer1.py --set_gpu='true' --prev_addr='' --prev_port=30011 --next_addr='10.96.0.212' --next_port=30012 --vram_limit=200 --debug=100 > GoogLeNetLayer1.txt;
python3 GoogLeNetLayer2.py --set_gpu='true' --prev_addr='' --prev_port=30012 --next_addr='10.96.0.213' --next_port=30013 --vram_limit=200 --debug=100 > GoogLeNetLayer2.txt;
python3 GoogLeNetLayer3.py --set_gpu='true' --prev_addr='' --prev_port=30013 --next_addr='10.96.0.200' --next_port=30010 --vram_limit=500 --debug=100 > GoogLeNetLayer3.txt;
python3 MobileNetLayer1.py --set_gpu='true' --prev_addr='' --prev_port=30021 --next_addr='10.96.0.222' --next_port=30022 --vram_limit=200 --debug=100 > MobileNetLayer1.txt;
python3 MobileNetLayer2.py --set_gpu='true' --prev_addr='' --prev_port=30022 --next_addr='10.96.0.223' --next_port=30023 --vram_limit=200 --debug=100 > MobileNetLayer2.txt;
python3 MobileNetLayer3.py --set_gpu='true' --prev_addr='' --prev_port=30023 --next_addr='10.96.0.200' --next_port=30020 --vram_limit=500 --debug=100 > MobileNetLayer3.txt;
python3 VGGNetLayer1.py --set_gpu='true' --prev_addr='' --prev_port=30031 --next_addr='10.96.0.232' --next_port=30032 --vram_limit=150 --debug=100 > VGGNetLayer1.txt;
python3 VGGNetLayer2.py --set_gpu='true' --prev_addr='' --prev_port=30032 --next_addr='10.96.0.233' --next_port=30033 --vram_limit=150 --debug=100 > VGGNetLayer2.txt;
python3 VGGNetLayer3.py --set_gpu='true' --prev_addr='' --prev_port=30033 --next_addr='10.96.0.200' --next_port=30030 --vram_limit=600 --debug=100 > VGGNetLayer3.txt;
python3 VGGFNetLayer1.py --set_gpu='true' --prev_addr='' --prev_port=30041 --next_addr='10.96.0.242' --next_port=30042 --vram_limit=200 --debug=100 > VGGFNetLayer1.txt;
python3 VGGFNetLayer2.py --set_gpu='true' --prev_addr='' --prev_port=30042 --next_addr='10.96.0.200' --next_port=30040 --vram_limit=500 --debug=100 > VGGFNetLayer2.txt;
sleep 3;
