import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# from warp_rnnt import rnnt_loss
import warprnnt_numba

import sentencepiece as spm
from jiwer import wer

from loguru import logger

from models.encoder import AudioEncoder
from models.decoder import Decoder
from models.jointer import Jointer

from constants import RNNT_BLANK, PAD, VOCAB_SIZE, TOKENIZER_MODEL_PATH, MAX_SYMBOLS
from constants import ATTENTION_CONTEXT_SIZE
from constants import N_STATE, N_LAYER, N_HEAD, N_MELS
from constants import BATCH_SIZE, NUM_WORKERS
from constants import MAX_EPOCHS, TOTAL_STEPS, WARMUP_STEPS, LR, MIN_LR
from constants import PRETRAINED_ENCODER_WEIGHT, TRAIN_MANIFEST, VAL_MANIFEST, LOG_DIR, BG_NOISE_PATH

from utils.dataset import AudioDataset, collate_fn
from utils.scheduler import WarmupLR

class StreamingRNNT(pl.LightningModule):
    def __init__(self, att_context_size, vocab_size, tokenizer_model_path):
        super().__init__()

        encoder_state_dict = torch.load(PRETRAINED_ENCODER_WEIGHT, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=True)
        # Create new keys 'conv3.weight', 'conv3.bias' that copy from 'conv2.weight', 'conv2.bias' so that we don't have to initialize conv3 weights
        encoder_state_dict['model_state_dict']['conv3.weight'] = encoder_state_dict['model_state_dict']['conv2.weight']
        encoder_state_dict['model_state_dict']['conv3.bias'] = encoder_state_dict['model_state_dict']['conv2.bias']

        self.encoder = AudioEncoder(
            n_mels=N_MELS,
            n_state=N_STATE,
            n_head=N_HEAD,
            n_layer=N_LAYER,
            att_context_size=att_context_size
        )
        self.encoder.load_state_dict(encoder_state_dict['model_state_dict'], strict=False)

        self.decoder = Decoder(vocab_size=vocab_size + 1)
        self.joint = Jointer(vocab_size=vocab_size + 1)

        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model_path)

        # self.loss = torchaudio.transforms.RNNTLoss(reduction="mean") # RNNTLoss has bug with logits number of elements > 2**31
        self.loss = warprnnt_numba.RNNTLossNumba(
            blank=RNNT_BLANK, reduction="mean", 
        )
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
        
    
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py#L416
    def greedy_decoding(self, x, x_len, max_symbols=None):
        enc_out, _ = self.encoder(x, x_len)
        all_sentences = []
        # greedy decoding, handle each sequence independently for easier implementation
        for batch_idx in range(enc_out.shape[0]):
            hypothesis = [[None, None]]  # [label, state]
            seq_enc_out = enc_out[batch_idx, :, :].unsqueeze(0) # [1, T, D]
            seq_ids = []
            
            for time_idx in range(seq_enc_out.shape[1]):
                curent_seq_enc_out = seq_enc_out[:, time_idx, :].unsqueeze(1) # 1, 1, D

                not_blank = True
                symbols_added = 0

                while not_blank and (max_symbols is None or symbols_added < max_symbols):
                    # In the first timestep, we initialize the network with RNNT Blank
                    # In later timesteps, we provide previous predicted label as input.
                    if hypothesis[-1][0] is None:
                        last_token = torch.tensor([[RNNT_BLANK]], dtype=torch.long, device=seq_enc_out.device)
                        last_seq_h_n = None
                    else:
                        last_token = hypothesis[-1][0]
                        last_seq_h_n = hypothesis[-1][1]
                    
                    if last_seq_h_n is None:
                        current_seq_dec_out, current_seq_h_n = self.decoder(last_token)
                    else:
                        current_seq_dec_out, current_seq_h_n = self.decoder(last_token, last_seq_h_n)
                    logits = self.joint(curent_seq_enc_out, current_seq_dec_out)[0, 0, 0, :]  # (B, T=1, U=1, V + 1)

                    del current_seq_dec_out

                    _, token_id = logits.max(0)
                    token_id = token_id.detach().item()  # K is the label at timestep t_s in inner loop, s >= 0.

                    del logits

                    if token_id == RNNT_BLANK:
                        not_blank = False
                    else:
                        symbols_added += 1
                        hypothesis.append([
                            torch.tensor([[token_id]], dtype=torch.long, device=curent_seq_enc_out.device),
                            current_seq_h_n
                        ])
                        seq_ids.append(token_id)
            all_sentences.append(self.tokenizer.decode(seq_ids))
        return all_sentences
    
    def process_batch(self, batch):
        return batch

    def training_step(self, batch, batch_idx):
        x, x_len, y, y_len = self.process_batch(batch)

        if batch_idx != 0 and batch_idx % 2000 == 0:
            all_pred = self.greedy_decoding(x, x_len, max_symbols=3)
            all_true = []
            for i, y_i in enumerate(y):
                y_i = y_i.cpu().numpy().astype(int).tolist()
                y_i = y_i[:y_len[i]]
                all_true.append(self.tokenizer.decode_ids(y_i))

            for pred, true in zip(all_pred, all_true):
                logger.debug(f"Pred: {pred}")
                logger.debug(f"True: {true}")
            
            all_wer = wer(all_true, all_pred)
            self.log("train_wer", all_wer, prog_bar=False, on_step=True, on_epoch=False)

        enc_out, x_len = self.encoder(x, x_len) # (B, T, Enc_dim)

        # Add a blank token to the beginning of the target sequence
        # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py#L185
        # Blank is also the start of sequence token and the sentence will start with blank; https://github.com/pytorch/audio/issues/3750
        y_start = torch.cat([torch.full((y.shape[0], 1), RNNT_BLANK, dtype=torch.int).to(y.device), y], dim=1).to(y.device)
        dec_out, _ = self.decoder(y_start) # (B, U, Dec_dim)
        logits = self.joint(enc_out, dec_out)

        input_lengths = x_len.int()
        target_lengths = y_len.int()
        targets = y.int()
        
        loss = self.loss(logits.to(torch.float32), targets, input_lengths, target_lengths)
        if batch_idx % 100 == 0:
            # Log training loss with format only two last digits
            self.log("train_loss", loss.detach().item(), prog_bar=True, on_step=True, on_epoch=False)
            # Log current learning rate
            self.log("lr", self.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, x_len, y, y_len = self.process_batch(batch)

        all_pred = self.greedy_decoding(x, x_len, max_symbols=3)
        all_true = []
        for i, y_i in enumerate(y):
            y_i = y_i.cpu().numpy().astype(int).tolist()
            y_i = y_i[:y_len[i]]
            all_true.append(self.tokenizer.decode_ids(y_i))
        
        all_wer = wer(all_true, all_pred)

        # ------------------CALCULATE LOSS------------------ 
        enc_out, x_len = self.encoder(x, x_len) # (B, T, Enc_dim)

        # Add a blank token to the beginning of the target sequence
        # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py#L185
        # Blank is also the start of sequence token and the sentence will start with blank; https://github.com/pytorch/audio/issues/3750
        y_start = torch.cat([torch.full((y.shape[0], 1), RNNT_BLANK, dtype=torch.int).to(y.device), y], dim=1).to(y.device)
        dec_out, _ = self.decoder(y_start) # (B, U, Dec_dim)
        logits = self.joint(enc_out, dec_out)

        input_lengths = x_len.int()
        target_lengths = y_len.int()
        targets = y.int()
        
        loss = self.loss(logits.to(torch.float32), targets, input_lengths, target_lengths)
        # ---------------------------------------------------

        if batch_idx % 1000 == 0:
            for pred, true in zip(all_pred, all_true):
                logger.debug(f"Pred: {pred}")
                logger.debug(f"True: {true}")

        self.log("val_loss", loss.item(), prog_bar=True)
        self.log("val_wer", all_wer, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
    # def on_validation_end(self):
    #     # save the last checkpoint
    #     self.trainer.save_checkpoint(f"{LOG_DIR}/rnnt-latest.ckpt", weights_only=True)
    #     return super().on_validation_end()

    def on_train_epoch_end(self):
        # save the last checkpoint
        self.trainer.save_checkpoint(f"{LOG_DIR}/rnnt-latest.ckpt", weights_only=True)
        return super().on_train_epoch_end()

    def configure_optimizers(self):
        return (
            [self.optimizer],
            # Change total steps to any number you want (should be total_samples/batch_size * epochs)
            [{"scheduler": WarmupLR(self.optimizer, WARMUP_STEPS, TOTAL_STEPS, MIN_LR), "interval": "step"}],
        )

train_dataset = AudioDataset(
    manifest_files=TRAIN_MANIFEST,
    bg_noise_path=BG_NOISE_PATH,
    shuffle=True,
    augment=True,
    tokenizer_model_path=TOKENIZER_MODEL_PATH
)

val_dataset = AudioDataset(
    manifest_files=VAL_MANIFEST,
    shuffle=False,
    tokenizer_model_path=TOKENIZER_MODEL_PATH
)
        
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True, collate_fn=collate_fn, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True, collate_fn=collate_fn, pin_memory=True)

model = StreamingRNNT(
    att_context_size=ATTENTION_CONTEXT_SIZE,
    vocab_size=VOCAB_SIZE,
    tokenizer_model_path=TOKENIZER_MODEL_PATH
)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    dirpath=LOG_DIR,
    filename='rnnt-{epoch:02d}-{val_loss:.2f}',
    save_top_k=2,
    mode='min'
)

trainer = pl.Trainer(
    # profiler="simple",
    # max_steps=100,
    max_epochs=MAX_EPOCHS, 
    devices=1,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    precision="bf16-mixed",
    strategy="ddp",
    callbacks=[checkpoint_callback],
    logger=pl.loggers.TensorBoardLogger(LOG_DIR),
    num_sanity_val_steps=0, # At the start, the model produced garbage predictions anyway. Should only be > 0 for testing
    check_val_every_n_epoch=2
)


# trainer.fit(model, train_dataloader, val_dataloader, ckpt_path="/path/to/ckpt")
trainer.fit(model, train_dataloader, val_dataloader)