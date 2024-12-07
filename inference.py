from fastapi import FastAPI, HTTPException, Query
import torch
import pandas as pd
from models.model import TwoTowerModel

# Initialize FastAPI app
app = FastAPI()

# Load model and data
user_input_dim = 11
EMBEDDING_DIM = 128

# Load necessary data
post_embed_df = pd.read_csv('post_embeddings.csv')
post_features_df = pd.read_csv('post_features_final.csv')
user_features_df = pd.read_csv('user_features_final.csv')
all_posts_df = pd.read_csv('all_posts_df.csv')

# Prepare post embeddings tensor
post_emb = torch.tensor(post_embed_df.drop(columns=['category_id', 'post_id']).values, dtype=torch.float32)


from transformers import BertTokenizer, BertModel

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


# Load the model
model = TwoTowerModel(user_input_dim, post_emb, EMBEDDING_DIM)
model.load_state_dict(torch.load('two_tower_model_final.pth'))
model.eval()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Helper function for recommendations
def get_recommendations(user_features, post_embeddings, top_k=10):
    user_features = torch.tensor(user_features, dtype=torch.float32).to(device)
    post_embeddings = torch.tensor(post_embeddings, dtype=torch.float32).to(device)

    with torch.no_grad():
        user_embedding = model.user_tower(user_features)
        scores = torch.matmul(user_embedding, post_embeddings.T)
        valid_k = min(top_k, scores.shape[1])
        top_k_indices = torch.topk(scores, k=valid_k, dim=1).indices.cpu().numpy()

    return top_k_indices


# API: Get recommendations (with optional category filter)
# @app.get("/feed")
# def get_feed(username: str, category_id: int = Query(None), top_k: int = 10):
#     """
#     API to fetch recommendations for a user.
#     - If `category_id` is provided, fetch category-specific recommendations.
#     - If `category_id` is not provided, fetch recommendations across all posts.
#     """
#     # Fetch user features
#     user_features = user_features_df.loc[user_features_df['username'] == username]
#     if user_features.empty:
#         raise HTTPException(status_code=404, detail="User not found")
#     user_features = user_features.drop(columns=['username', 'user_id']).values

#     if category_id is not None:
#         # Filter post features by category
#         category_posts = post_embed_df.loc[post_embed_df['category_id'] == category_id]
#         if category_posts.empty:
#             raise HTTPException(status_code=404, detail="invalid category id")
#         post_features = category_posts.drop(columns=['category_id', 'post_id']).values
#         recommendations = get_recommendations(user_features, post_features, top_k)
#         final_rec = post_features_df.loc[post_features_df['category_id'] == category_id].iloc[recommendations.squeeze(0),:].index
#     else:
#         # Use all post features
#         post_features = post_embed_df.drop(columns=['category_id', 'post_id']).values
#         recommendations = get_recommendations(user_features, post_features, top_k)
#         final_rec = post_features_df.iloc[recommendations.squeeze(0), :].index

#     # Fetch recommended video links
#     rec_video_links = all_posts_df.iloc[final_rec,:]["video_link"].values.tolist()
#     return {"username": username, "category_id": category_id, "video_links": rec_video_links}


@app.get("/feed")
def get_feed(username: str, category_id: int = Query(None), mood: str = Query(None), top_k: int = 10):
    """
    API to fetch recommendations for a user.
    - If `category_id` is provided, fetch category-specific recommendations.
    - If `category_id` and `mood` are provided, fetch mood-specific category recommendations.
    - If neither `category_id` nor `mood` is provided, fetch general recommendations across all posts.
    """
    # Fetch user features
    user_features = user_features_df.loc[user_features_df['username'] == username]
    if user_features.empty:
        raise HTTPException(status_code=404, detail="User not found")
    user_features = user_features.drop(columns=['username', 'user_id']).values

    # Handle mood embedding if mood is provided
    if mood:
        mood_emb = bert_model(torch.tensor([tokenizer.encode(mood, add_special_tokens=True)])).last_hidden_state.squeeze(0)[0]
        mood_emb = torch.tensor(mood_emb.detach().numpy(), dtype=torch.float32).unsqueeze(0)
        all_post_mood_emb = torch.tensor(post_features_df.iloc[:, -768:].values, dtype=torch.float32)
        scores = torch.matmul(mood_emb, all_post_mood_emb.T)
        top_k = min(150, scores.shape[1])
        top_k_indices = torch.topk(scores, k=top_k, dim=1).indices.cpu().numpy()
    
    if category_id is not None:
        # If mood is provided, filter posts based on both category and mood
        if mood:
            category_posts = post_embed_df.loc[top_k_indices[0]].loc[post_embed_df['category_id'] == category_id]
            post_features = category_posts.drop(columns=['category_id', 'post_id']).values
            recommendations = get_recommendations(user_features, post_features, top_k)
            final_rec = post_features_df.loc[post_features_df['category_id'] == category_id].iloc[recommendations.squeeze(0), :].index
        else:
            # If only category_id is provided, filter posts based on category
            category_posts = post_embed_df.loc[post_embed_df['category_id'] == category_id]
            if category_posts.empty:
                raise HTTPException(status_code=404, detail="Invalid category id")
            post_features = category_posts.drop(columns=['category_id', 'post_id']).values
            recommendations = get_recommendations(user_features, post_features, top_k)
            final_rec = post_features_df.loc[post_features_df['category_id'] == category_id].iloc[recommendations.squeeze(0), :].index
    else:
        # If no category_id is provided, use all posts
        post_features = post_embed_df.drop(columns=['category_id', 'post_id']).values
        recommendations = get_recommendations(user_features, post_features, top_k)
        final_rec = post_features_df.iloc[recommendations.squeeze(0), :].index

    # Fetch recommended video links
    rec_video_links = all_posts_df.iloc[final_rec, :]["video_link"].values.tolist()
    return {"username": username, "category_id": category_id, "mood": mood, "video_links": rec_video_links}
