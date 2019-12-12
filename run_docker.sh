# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# docker pull anibali/pytorch:cuda-10.0
cd docker-pytorch/cuda-10.0
docker build -t rgn_pytorch:cuda_10.0
cd ../..

DATA_LOCATION=$1
WORKING_FOLDER=$2
if [ WORKING_FOLDER == "" ]; then
    WORKING_FOLDER=$PWD
fi
docker run --rm -it --init \
  --runtime=nvidia \
  --ipc=host \
  --user="$(id -u):$(id -g)" \
  --volume="$WORKING_FOLDER:/app" \ # Mount current dir
  --volume="$DATA_LOCATION:/data" \ # Mount data
  -e NVIDIA_VISIBLE_DEVICES=0 \
  anibali/pytorch python3 model_test_server.py


