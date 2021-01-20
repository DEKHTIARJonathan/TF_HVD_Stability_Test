docker build -t horovod_test_container:latest .

docker run -it --rm \
  --gpus '"device=0,1"' \
  --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd):/workspace \
  horovod_test_container:latest bash -c "python3.8 -m pytest"
