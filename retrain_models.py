"""
retrain_models.py  —  SMS-only version
Run from project root (same folder as manage.py):
    python retrain_models.py
"""

import pickle
import string
import pandas as pd
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# ── text_process (same as utils.py) ──────────────────────────────────────────
def text_process(message):
    noPunc = [char for char in message if char not in string.punctuation]
    noPunc = ''.join(noPunc)
    return [word for word in noPunc.split() if word not in stopwords.words('english')]

# ── Load SMS dataset ──────────────────────────────────────────────────────────
print("Loading SMS data...")
df = pd.read_csv(
    'machine_learning_section/smsspamcollection/SMSSpamCollection',
    sep='\t', names=['labels', 'message']
)
print(f"Loaded {len(df)} rows  |  spam: {(df['labels']=='spam').sum()}  ham: {(df['labels']=='ham').sum()}")

# ── Preprocess ────────────────────────────────────────────────────────────────
print("Preprocessing...")
df['message'] = df['message'].apply(text_process)
df['message'] = df['message'].apply(lambda x: ' '.join(x))

# ── Train Pipeline 1 ──────────────────────────────────────────────────────────
print("Training pipeline 1...")
pipeline = Pipeline([
    ('vec',   CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf',   MultinomialNB(alpha=0.1)),
])
pipeline.fit(df['message'], df['labels'])
with open('text_clf_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print("  Saved  text_clf_pipeline.pkl")

# ── Train Pipeline 2 ──────────────────────────────────────────────────────────
print("Training pipeline 2...")
pipeline_second = Pipeline([
    ('vec',   CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf',   MultinomialNB(alpha=0.1)),
])
pipeline_second.fit(df['message'], df['labels'])
with open('spam_clf_model_pipeline_final_second.pkl', 'wb') as f:
    pickle.dump(pipeline_second, f)
print("  Saved  spam_clf_model_pipeline_final_second.pkl")

print("\nDone!  Run:  python manage.py runserver")