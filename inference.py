import torch
import torch.nn as nn
import sys
sys.path.append('models')

from model import TwoTowerModel    
import pandas as pd

user_input_dim = 11
EMBEDDING_DIM = 128
post_embed_df = pd.read_csv('post_features_final.csv')
post_emb = torch.tensor(post_embed_df.values, dtype = torch.float32).squeeze(0)
model = TwoTowerModel(user_input_dim, post_emb, EMBEDDING_DIM)
model.load_state_dict(torch.load('two_tower_model_final.pth', weights_only=True))

post_features_df = pd.read_csv('post_features_final.csv')
user_features_df = pd.read_csv('user_features_final.csv')
all_posts_df = pd.read_csv('all_posts_df.csv')
post_embed_df = pd.read_csv('post_embeddings.csv')

def get_recommendations(model, user_features, post_embeddings, top_k=10):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    user_features = torch.tensor(user_features, dtype=torch.float32).to(device)
    post_embeddings = torch.tensor(post_embeddings, dtype=torch.float32).to(device)

    with torch.no_grad():
        user_embedding = model.user_tower(user_features)

        scores = torch.matmul(user_embedding, post_embeddings.T)
        valid_k = min(top_k, scores.shape[1])
        top_k_indices = torch.topk(scores, k=valid_k, dim=1).indices.cpu().numpy()

    return top_k_indices

user_name = "doey" #input("Enter the username: ")
category_id = 2 #int(input("Enter the category id: "))
user_features = user_features_df.loc[user_features_df['username'] == user_name].drop(columns=['username', 'user_id']).values
if category_id:
  post_features = post_embed_df.loc[post_embed_df['category_id'] == category_id].drop(columns=['category_id','post_id']).values
  recommendations = get_recommendations(model, user_features, post_features)
  final_rec = post_features_df.loc[post_features_df['category_id'] == category_id].iloc[recommendations.squeeze(0),:].index

else:
  post_features = post_embed_df.iloc[:, 0:].drop(columns=['category_id','post_id']).values      # All posts
  recommendations = get_recommendations(model, user_features, post_features)
  final_rec = post_features_df.iloc[:,recommendations.squeeze(0)]['post_id']
if post_features.shape[0]==0:
  print("No posts found")
  exit()

rec_video_links = all_posts_df.iloc[final_rec,:]["video_link"].values
print(rec_video_links)