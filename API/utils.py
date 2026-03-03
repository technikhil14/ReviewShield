import re
import pandas as pd

def engineer_features(text: str, rating: float) -> dict:
    
    review_length = len(text)
    words = text.split()
    word_count = len(words)
    
    sentence_count = len(re.split(r'[.!?]+', text.strip()))
    sentence_count = max(sentence_count, 1)
    
    avg_sentence_length = review_length / sentence_count
    
    raw_unique_ratio = len(set(w.lower() for w in words)) / (word_count + 1)
    
    if review_length < 150:
        unique_word_ratio = raw_unique_ratio * (review_length / 150)
    else:
        unique_word_ratio = raw_unique_ratio
    
    has_digits = int(any(c.isdigit() for c in text))
    
    return {
        "unique_word_ratio": unique_word_ratio,
        "avg_sentence_length": avg_sentence_length,
        "review_length": review_length,
        "has_digits": has_digits
    }


def features_to_df(features: dict) -> pd.DataFrame:
    return pd.DataFrame([features])