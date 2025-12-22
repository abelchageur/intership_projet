from fastapi import FastAPI
import pickle, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pydantic import BaseModel

app = FastAPI()

MODEL_DIR = "/app/models/"

# ------------------------------
# Load all saved models + data
# ------------------------------
bst = lgb.Booster(model_file=MODEL_DIR + "lgb_product_demand.txt")
tfv = pickle.load(open(MODEL_DIR + "tfv.pkl", "rb"))
lr = pickle.load(open(MODEL_DIR + "lr.pkl", "rb"))
scaler = pickle.load(open(MODEL_DIR + "scaler.pkl", "rb"))
oenc = pickle.load(open(MODEL_DIR + "oenc.pkl", "rb"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load training embeddings + texts + labels
train_emb = np.load(MODEL_DIR + "train_emb.npy")
train_texts = pickle.load(open(MODEL_DIR + "train_texts.pkl", "rb"))
y_train = np.load(MODEL_DIR + "y_train.npy")


class ProductInput(BaseModel):
    name: str
    description: str
    Brand: str
    Material: str
    Color: str
    price: float


@app.post("/predict")
def predict(data: ProductInput):

    df = pd.DataFrame([data.dict()])
    text = df['name'][0] + ". " + df['description'][0]

    # ------------------------------
    # Preprocess for prediction
    # ------------------------------
    num = scaler.transform(df[['price']])
    cat = oenc.transform(df[['Brand', 'Material', 'Color']])
    emb = embedder.encode([text], show_progress_bar=False)

    X_new = np.hstack([emb, num, cat])

    # ------------------------------
    # Main demand prediction
    # ------------------------------
    demand_prob = float(bst.predict(X_new)[0])
    demand_class = 1 if demand_prob >= 0.5 else 0

    # ------------------------------
    # Global TF-IDF keyword scoring
    # ------------------------------
    tfidf_vec = tfv.transform([text])
    coef = lr.coef_[0]
    feat_names = tfv.get_feature_names_out()

    contrib = tfidf_vec.multiply(coef)
    vals = np.array(contrib.sum(axis=0)).flatten()

    top_idx = vals.argsort()[::-1][:10]
    top_keywords = [(feat_names[i], float(vals[i])) for i in top_idx]

    # ============================================================
    # CATEGORY-AWARE KEYWORD SUGGESTION
    # ============================================================

    # (1) Find similar products using cosine similarity on embeddings
    similarities = cosine_similarity(emb, train_emb)[0]

    N = 200
    top_idx_sim = similarities.argsort()[::-1][:N]

    local_train_texts = [train_texts[i] for i in top_idx_sim]
    local_y = y_train[top_idx_sim]

    # (2) TF-IDF only on similar products
    local_tfv = TfidfVectorizer(max_features=5000,
                                ngram_range=(1,2),
                                stop_words='english')
    local_tfidf = local_tfv.fit_transform(local_train_texts)
    local_feat_names = local_tfv.get_feature_names_out()

    # (3) Train logistic regression only on the similar group
    local_lr = LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        C=0.5
    )
    local_lr.fit(local_tfidf, local_y)

    local_coef = local_lr.coef_[0]

    # (4) Top 200 category keywords
    local_pos = sorted(zip(local_coef, local_feat_names), reverse=True)
    local_positive_keywords = [w for _, w in local_pos[:200]]

    # (5) Which keywords the new product does NOT contain
    local_product_tfidf = local_tfv.transform([text])
    product_present_idx = local_product_tfidf.nonzero()[1]
    product_words = {local_feat_names[i] for i in product_present_idx}

    missing_keywords = [w for w in local_positive_keywords if w not in product_words]
    category_missing_top10 = missing_keywords[:10]

    # ------------------------------
    # Return full structured data
    # ------------------------------
    return {
        "shape": list(X_new.shape),
        "demand_prob": demand_prob,
        "demand_class": demand_class,
        "top_keywords": top_keywords,
        "category_missing_keywords": category_missing_top10
    }
