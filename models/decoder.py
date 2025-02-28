from torch import nn
import torch

class Decoder(nn.Module):
    def __init__(self, vocab_size=1024, embed_dim=768, hidden_dim=768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
    
    def forward(self, y, h_0=None):
        y = self.embedding(y)  # (B, U, Embed_dim)
        if h_0 is not None:
            y, h_n = self.rnn(y, h_0)
        else:
            # random state sampling: https://arxiv.org/pdf/1910.11455
            h_0 = torch.randn(1, y.size(0), self.hidden_dim, device=y.device)
            y, h_n = self.rnn(y, h_0)  # (B, U, Hidden_dim)
        return y, h_n