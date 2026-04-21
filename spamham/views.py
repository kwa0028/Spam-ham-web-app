from django.shortcuts import render
import numpy as np
import pickle
import string
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from utils import text_process

# Load once at startup, not on every request
with open('text_clf_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)
with open('spam_clf_model_pipeline_final_second.pkl', 'rb') as f:
    pipeline_second = pickle.load(f)

def get_token_heatmap(original_message, is_spam):
    """
    Returns a list of (word, intensity) tuples for the original message words.
    intensity is 0.0-1.0: for spam, high = spammy; for ham, high = hammy.
    Words not in the vocabulary get intensity 0.0.
    """
    vec = pipeline.named_steps['vec']
    clf = pipeline.named_steps['clf']
 
    spam_class_idx = list(clf.classes_).index('spam')
    ham_class_idx = 1 - spam_class_idx
    spam_log_probs = clf.feature_log_prob_[spam_class_idx]
    ham_log_probs = clf.feature_log_prob_[ham_class_idx]
    log_odds = spam_log_probs - ham_log_probs  # positive = spammy, negative = hammy
 
    feature_names = vec.get_feature_names_out()
    token_to_score = {feat: score for feat, score in zip(feature_names, log_odds)}
 
    words = original_message.split()
    scored_words = []
    for word in words:
        clean = ''.join(ch for ch in word if ch not in string.punctuation).lower()
        score = token_to_score.get(clean, 0.0)
        scored_words.append((word, float(score)))
 
    scores = np.array([s for _, s in scored_words])
 
    if is_spam:
        pos_scores = np.maximum(scores, 0)
        max_val = pos_scores.max() if pos_scores.max() > 0 else 1.0
        intensities = pos_scores / max_val
    else:
        neg_scores = np.maximum(-scores, 0)
        max_val = neg_scores.max() if neg_scores.max() > 0 else 1.0
        intensities = neg_scores / max_val
 
    return [(word, round(float(intensity), 3)) for (word, _), intensity in zip(scored_words, intensities)]
 

# Create your views here.
def home(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        message_cp = message
        message_list = [message]
        processed = text_process(message_list)
        processed_joined = [' '.join(processed)]
        result, accuracy = predict(processed_joined)
        is_spam = 'spam' in result
        heatmap_tokens = get_token_heatmap(message_cp, is_spam)
        return render(request, 'home.html', {
            'result': result,
            'message': message_cp,
            'accuracy': accuracy,
            'heatmap_tokens': heatmap_tokens,
            'is_spam': is_spam,
        })
 
    return render(request, 'home.html')
 
 
def predict(message):
    result = " "
 
    test = pipeline.predict(message)
    test_prob = pipeline.predict_proba(message)
    test_second = pipeline_second.predict(message)
    test_second_prob = pipeline_second.predict_proba(message)
 
    value_spam = test_prob[0][1]
    value_spam_second = test_second_prob[0][1]
    value_ham = test_prob[0][0]
    value_ham_second = test_second_prob[0][0]
 
    if value_spam > 0.5 and value_spam_second > 0.5:
        result = 'spam'
        accuracy = max(value_spam, value_spam_second)
    elif value_spam <= 0.5 and value_spam_second <= 0.5:
        result = 'ham'
        accuracy = max(value_ham, value_ham_second)
    elif value_spam > 0.5 or value_spam_second > 0.5:
        if max(value_spam, value_spam_second) + 0.1 > max(value_ham, value_ham_second):
            accuracy = max(value_spam, value_spam_second)
            result = 'spam'
        else:
            result = 'ham'
            accuracy = max(value_ham, value_ham_second)
    else:
        result = 'ham'
        accuracy = max(value_ham, value_ham_second)
 
    accuracy = round(accuracy, 2) * 100
 
    if result == 'spam':
        result = 'very likely a spam' if accuracy > 80 else 'less likely a spam'
    elif accuracy > 80:
        result = 'very likely a ham'
    else:
        result = 'less likely a ham'
 
    return result, accuracy