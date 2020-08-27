docker build -t horovod_test_container:latest .

docker run -it --rm \
  --gpus all \
  --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
  horovod_test_container:latest bash -c "python3.7 -m pytest"
