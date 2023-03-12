import joblib
from model import inference
import numpy as np

def online_inference(row_dict, cat_features):
    # load the model from `model_path`
    model = joblib.load('./model/model.pkl')
    encoder = joblib.load('./model/encoder.pkl')

    row_transformed = list()
    X_categorical = list()
    X_continuous = list()

    for key, value in row_dict.items():
        mod_key = key.replace('_', '-')
        if mod_key in cat_features:
            X_categorical.append(value)
        else:
            X_continuous.append(value)

    y_cat = encoder.transform([X_categorical])
    y_conts = np.asarray([X_continuous])

    row_transformed = np.concatenate([y_conts, y_cat], axis=1)

    # get inference from model
    preds = inference(model=model, X=row_transformed)

    return 'hi_income' if preds[0] else 'low_income'