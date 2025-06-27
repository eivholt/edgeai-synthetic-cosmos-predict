https://docs.lambda.ai/public-cloud/guest-agent/

ubuntu@209-20-157-175:/mnt$ df -h -BG
Filesystem                           1G-blocks  Used Available Use% Mounted on
tmpfs                                      23G    1G       23G   1% /run
/dev/vda1                                 993G   25G      968G   3% /
tmpfs                                     111G    1G      111G   1% /dev/shm
tmpfs                                       1G    0G        1G   0% /run/lock
/dev/vda15                                  1G    1G        1G   6% /boot/efi
f51494e3-f70c-4498-a376-e4b3f9eab023  2100036G    0G  2100036G   0% /home/ubuntu/cosmos-utah


echo "nvapi-" | docker login nvcr.io \
  --username '$oauthtoken' \
  --password-stdin

docker pull nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.0

pip install --upgrade huggingface_hub
huggingface-cli login

git clone https://github.com/nvidia-cosmos/cosmos-predict2.git

docker run --gpus all -it --rm \
-v /path/to/cosmos-predict2:/workspace \
-v /path/to/datasets:/workspace/datasets \
-v /path/to/checkpoints:/workspace/checkpoints \
nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.0

docker run --gpus all -it --rm \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/datasets":/workspace/datasets \
  -v "$(pwd)/checkpoints":/workspace/checkpoints \
  nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.0

python -m scripts.download_checkpoints --model_sizes 2B --model_types text2image --checkpoint_dir checkpoints
