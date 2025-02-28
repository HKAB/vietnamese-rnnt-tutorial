from torch import nn

class Jointer(nn.Module):
    def __init__(self, encoder_dim=768, decoder_dim=768, vocab_size=1024):
        super().__init__()
        # It should be noted that the encoder_dim and decoder_dim can be different
        # In that case, we can use a linear layer to project the encoder_dim to decoder_dim
        # self.encoder_projection = nn.Linear(encoder_dim, decoder_dim)
        self.fc = nn.Linear(encoder_dim, vocab_size)
    
    def forward(self, enc, dec):
        # https://github.com/NVIDIA/NeMo/blob/066e4b4f1bdb38b2010de1ef950f42e9dddcd951/nemo/collections/asr/modules/rnnt.py#L1523
        enc = enc.unsqueeze(2) # (B, T, 1, H)
        dec = dec.unsqueeze(1) # (B, 1, U, H)
        combined = enc + dec  # (B, T, U, H)
        
        return self.fc(combined)  # (B, T, U, Vocab_size)