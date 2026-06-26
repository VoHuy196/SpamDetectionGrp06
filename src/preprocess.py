import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess(df: pd.DataFrame, test_size: float, random_state: int, max_features: int):
    df = df.dropna().drop_duplicates()
    X = df['message']
    y = df['label']
    
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tr_vec = vectorizer.fit_transform(X_tr).toarray()
    X_te_vec = vectorizer.transform(X_te).toarray()
    
    return X_tr_vec, X_te_vec, y_tr.values, y_te.values, vectorizer
