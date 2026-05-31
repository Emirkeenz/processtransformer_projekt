import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceActivityPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, time_hidden_dim=1):
        super(SequenceActivityPredictor, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        
        self.W_z = nn.Linear(embed_dim + hidden_dim, hidden_dim)
        self.W_r = nn.Linear(embed_dim + hidden_dim, hidden_dim)
        self.W_h = nn.Linear(embed_dim + hidden_dim, hidden_dim)
        
        self.activity_output = nn.Linear(hidden_dim, vocab_size)
        self.time_output = nn.Linear(hidden_dim, 1)
    
    def forward(self, activity_indices):
        batch_size, seq_len = activity_indices.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=activity_indices.device)
        embedded = self.embedding(activity_indices)
        
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            concat = torch.cat([x_t, h], dim=1)
            z = torch.sigmoid(self.W_z(concat))
            r = torch.sigmoid(self.W_r(concat))
            r_concat = torch.cat([r * x_t, h], dim=1)
            h_tilde = torch.tanh(self.W_h(r_concat))
            h = z * h + (1 - z) * h_tilde
        
        activity_logits = self.activity_output(h)
        time_prediction = self.time_output(h)
        
        return activity_logits, time_prediction