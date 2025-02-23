# Streaming RNN-T with Whisper Encoder

This is a tutorial on how to train your own streaming RNN-T model using state-of-the-art encoder. Instead of using complex framework, we'll implement it from scratch and using pytorch lighting to train.

This repo contain:
- Tutorial notebooks
- Training script
- Streaming inference script
- ONNX export script

We also share a streaming RNN-T model trained on 7000 Vietnamese hours (1000 labeled hours and 5000 hours labeled by Whisper-v3-turbo)
## Docker

I use `torch.nn.functional.scaled_dot_product_attention` so it's recommend to use version of torch that support it.

```bash
docker build -t pl25_rnnt .

docker run -itd --gpus all --net host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name YOUR_DOCKER_NAME -v localpath:/wp pl25_rnnt
```

## Usage

### Training
1. Prepare train, validation manifest in NeMo format. Put them into `TRAIN_MANIFEST` and `VAL_MANIFEST`.
2. Prepare background noise for audio augmentation. There are three data you might use: [AudioSet](https://research.google.com/audioset/download.html), [MUSAN](https://www.openslr.org/17/) and [FSDnoisy18k](https://zenodo.org/records/2529934). Download them and put their path into `BG_NOISE_PATH`.
3. Download whisper encoder weight from Huggingface and put it into `PRETRAINED_ENCODER_WEIGHT`.
4. Change any parameters related to optimizer, scheduler, batch size, num workers,...

### Inference & ONNX export
Download the pretrained weight from Huggingface or use your own weight.

## Contributing

Any contributions are welcomed.

## License

[Apache license 2.0](https://www.apache.org/licenses/LICENSE-2.0)