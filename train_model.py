# Script to train machine learning model.
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from data import process_data
from model import train_model, compute_model_metrics, inference

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


# Add code to load in the data.
logging.info("loading data...")
path = './data/cleaned_census.csv'
data = pd.read_csv(path)
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]

# Proces the train and test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=cat_features,
        label='salary',
        training=False,
        encoder=encoder,
        lb=lb
    )

# Train and save a model.
logging.info("training model")

model = train_model(X_train, y_train)

# Scoring
logging.info("evaluating model")
y_preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_preds)
logging.info(f"Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}")

# Save artifacts
logging.info("Saving model artifacts")
joblib.dump(model, './model/model.pkl')
joblib.dump(encoder, './model/encoder.pkl')
joblib.dump(lb, './model/lb.pkl')
