import torch
import torch.nn as nn

import pandas as pd

class TwoTowerModel(nn.Module):
    def __init__(self, user_input_dim, post_emb, embedding_dim):
        super(TwoTowerModel, self).__init__()
        # User tower
        self.post_embedding = post_emb
        self.user_tower = nn.Sequential(
            nn.Linear(user_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, user_features):
        user_embedding = self.user_tower(user_features)
        return user_embedding

    def predict(self, user_features, post_embeddings):
        user_embedding = self.forward(user_features)
        return torch.sum(user_embedding * post_embeddings, dim=-1)