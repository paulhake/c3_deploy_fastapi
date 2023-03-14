import joblib
import numpy as np
from model import inference


def online_inference(row_dict, cat_features):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    row_dict : json format
        model input features for inference
    cat_features : list
        List of categorical features for data preprocessing
    Returns
    -------
    preds : text value
        Predictions from the model - hi_income or low_income
    """
    # load the model
    model = joblib.load('./model/model.pkl')
    # load encoder
    encoder = joblib.load('./model/encoder.pkl')

    row_transformed = []
    X_categorical = []
    X_continuous = []
    # iterate through input dictionary and group as categorical or continuous
    # features
    for key, value in row_dict.items():
        mod_key = key.replace('_', '-')
        if mod_key in cat_features:
            X_categorical.append(value)
        else:
            X_continuous.append(value)
    # encode cat features to one hot
    y_cat = encoder.transform([X_categorical])
    # format continuous features as array
    y_conts = np.asarray([X_continuous])
    # concat features to single array for inference
    row_transformed = np.concatenate([y_conts, y_cat], axis=1)

    # get inference from model
    preds = inference(model=model, X=row_transformed)
    # return text indicators for hi or low income
    return 'hi_income' if preds[0] else 'low_income'
