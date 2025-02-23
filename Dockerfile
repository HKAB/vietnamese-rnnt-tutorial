FROM pytorchlightning/pytorch_lightning:2.5.0-py3.11-torch2.2-cuda12.1.1

RUN pip install --upgrade git+https://github.com/titu1994/warprnnt_numba.git

RUN pip install sentencepiece jiwer loguru audiomentations

# for warprnnt_numba
RUN apt-get update && apt-get install -y cuda-toolkit-12-1 