import torch
import torch.nn as nn

# creating a class for two tower architecture
class TwoTowerModel(nn.Module):
    '''Two Tower Model that takes user features and computes user embeddings.
    
    This model is typically used for recommendation systems where the two towers 
    (user and post) are used to generate embeddings for users and posts separately, 
    and then compare those embeddings (e.g., through dot product or cosine similarity) 
    to make predictions.
    '''
    def __init__(self, user_input_dim, post_emb, embedding_dim):
        '''
        Initializes the TwoTowerModel.

        Args:
            user_input_dim (int): The dimensionality of the input features for the user.
            post_emb (Tensor): Pre-trained or initialized post embeddings (typically of shape [num_posts, embedding_dim]).
            embedding_dim (int): The dimensionality of the embeddings for both user and post representations.
        '''
        super(TwoTowerModel, self).__init__()
        # User tower
        self.post_embedding = post_emb
        self.user_tower = nn.Sequential(
            nn.Linear(user_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, user_features):
        ''' 
        Computes the user embedding based on the input user features.

        Args:
            user_features (Tensor): The input features for the user (e.g., a tensor of shape [batch_size, user_input_dim]).

        Returns:
            Tensor: The computed user embedding (of shape [batch_size, embedding_dim]).
        '''
        user_embedding = self.user_tower(user_features)
        return user_embedding

    def predict(self, user_features, post_embeddings):
         ''' 
        Computes the similarity between the user embedding and post embeddings. 

        This method calculates the dot product between the user embedding and each 
        post embedding, which is often used for predicting user preferences for posts.

        Args:
            user_features (Tensor): The input features for the user (e.g., a tensor of shape [batch_size, user_input_dim]).
            post_embeddings (Tensor): The embeddings for posts (e.g., a tensor of shape [num_posts, embedding_dim]).

        Returns:
            Tensor: A tensor of similarity scores (e.g., dot products) between the user embedding and each post embedding.
        '''
        user_embedding = self.forward(user_features)
        return torch.sum(user_embedding * post_embeddings, dim=-1)
    
