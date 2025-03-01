# Vietnamese Streaming RNN-T

Welcome to the tutorial on training your own streaming RNN-T model using a state-of-the-art encoder! In this guide, we'll walk you through the process of building a "powerful" model from scratch, without relying on complex frameworks.

<div align="center">
<img src="./media/Intro.png" alt="Efficient minimum word error rate training of RNN-transducer for end-to-end speech recognition">
<i>Visualization of RNN-T</i>
</div>
<br>

- **Demo 🤖**: [Check out the demo on Huggingface Spaces](https://huggingface.co/spaces/hkab/rnnt-whisper-encoder)
- **Blog 📃**: [Read the detailed blog post](https://hkab.substack.com/publish/post/157867185)

## 🗂️ What's Inside

This repository includes:
- Comprehensive training scripts
- Streaming inference scripts
- ONNX export scripts

🌟 We also provide a pre-trained streaming RNN-T model, trained on 6000 hours of Vietnamese audio (1000 labeled hours and 5000 hours labeled by Whisper-v3-turbo). If you only care about the inference part, two notebooks in the `./notebooks/` folder are all you need.

## 🐋 Docker Setup

To ensure compatibility, we recommend using a version of PyTorch that supports `torch.nn.functional.scaled_dot_product_attention`.

```bash
git clone https://github.com/HKAB/rnnt-whisper-tutorial.git

cd rnnt-whisper-tutorial

docker build -t pl25_rnnt .

docker run -itd --gpus all --net host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name YOUR_DOCKER_NAME -v /path/to/local:/wp pl25_rnnt
```

## ⚙️ Usage

### 🏋️ Training (Not recommend)

Training an RNN-T model is highly GPU-intensive. To train using this repository with `BATCH_SIZE` of 32 and audio lengths of approximately 15 seconds, you will need around **50GB** of vRAM. Here are steps required before training:

1. **Prepare Manifests, Tokenizer**: Create training and validation manifests in NeMo format (there is a sample at `./data/sample.jsonl`) and set their paths to `TRAIN_MANIFEST` and `VAL_MANIFEST`. Then prepare a tokenizer on your text using [sentencepiece](https://github.com/google/sentencepiece).
2. **Prepare Background Noise for Augmentation**: Enhance your training data with background noise from sources like [AudioSet](https://research.google.com/audioset/download.html), [MUSAN](https://www.openslr.org/17/), and [FSDnoisy18k](https://zenodo.org/records/2529934). Set the path to these datasets in `BG_NOISE_PATH`.
3. **Get Pretrained Encoder Weights**: Download the Whisper weights from [here](https://github.com/openai/whisper/blob/main/whisper/__init__.py) to `./weights` and then run `python3 export_encoder.py` to extract the encoder weight for our use. Finally, set the encoder path to `PRETRAINED_ENCODER_WEIGHT`.
4. **Adjust Parameters**: Customize parameters related to the optimizer, scheduler, batch size, number of workers, and more to suit your needs in `constants.py`. 

### ⚡ Inference & ONNX Export

Two notebooks in the `notebooks` folder will guide you through the process.

## 🤝 Contributing

We welcome any contributions! Feel free to open issues or submit pull requests to improve this project.

## ⚖️ License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).