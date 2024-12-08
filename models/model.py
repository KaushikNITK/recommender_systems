import torch
import torch.nn as nn

# creating a class for two tower architecture
class TwoTowerModel(nn.Module):
    '''Two Tower Model takes in user features and gives out user embeddings'''
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
        '''this derives embeddings of user based on the parameters learnt form the training dataset'''
        user_embedding = self.user_tower(user_features)
        return user_embedding

    def predict(self, user_features, post_embeddings):
        ''' this function outputs the dot product or cosine similarities of user features with post features'''
        user_embedding = self.forward(user_features)
        return torch.sum(user_embedding * post_embeddings, dim=-1)
    
