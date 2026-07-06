import pickle
import pandas as pd
from preprocessing import clean_data, encode_features
from config import MODEL_PATH, TARGET_COL, TARGET_CLASSES


def load_model(path=MODEL_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)


def align_columns(df, model):
    if hasattr(model, 'feature_names_in_'):
        cols = [c for c in model.feature_names_in_ if c in df.columns]
        return df[cols]
    return df


def predict_single(model, input_dict):
    """
    Predict financial health category for a single business.
    Returns dict with prediction (int), label (Low/Medium/High), probabilities.
    """
    # TODO: DataFrame, clean, encode, align, predict
    df = pd.DataFrame([input_dict])
    df = clean_data(df)
    df = encode_features(df)
    df = align_columns(df, model)
    # TODO: return {'prediction': int, 'label': TARGET_CLASSES[prediction],
    #               'probabilities': dict of class probabilities}
    prediction = int(model.predict(df)[0])
    label = TARGET_CLASSES[prediction]

    probabilities = {}
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(df)[0]
        probabilities = {TARGET_CLASSES[i]: round(float(p), 4) for i, p in enumerate(proba)}

    return {
        'prediction': prediction,
        'label': label,
        'probabilities': probabilities
    }
   


def predict_batch(model, df):
    """
    Predict financial health for a batch of businesses.
    Returns df with Prediction, Label columns added.
    """
    # TODO: copy, clean, encode, align, predict
    result = df.copy()
    cleaned = clean_data(result)
    encoded = encode_features(cleaned)
    aligned = align_columns(encoded, model)

    predictions = model.predict(aligned)
    result['Prediction'] = predictions
    result['Label'] = [TARGET_CLASSES[p] for p in predictions]

    return result
