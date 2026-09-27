"""
Central configuration for all four detectors.

Drop your CSVs into backend/data/ using these filenames (or edit the
paths below to match your actual filenames):

    data/sms_scam.csv          -> SMS scam/spam text dataset
    data/url_scam.csv          -> malicious URL dataset
    data/bank_fraud.csv        -> bank transaction fraud dataset
    data/credit_card_fraud.csv -> credit card fraud dataset (e.g. Kaggle's V1..V28 dataset)

Column names vary a lot between public datasets, so each config lists
several common aliases. The loader (data_utils.py) picks the first
alias that exists in your file. If none match, edit the alias list
here to add your dataset's exact column name -- that's the only
change you should need to make.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATASETS = {
    "sms": {
        "path": os.path.join(DATA_DIR, "sms_spam_collection.csv"),
        "type": "text",
        "text_col_aliases": ["text", "message", "sms", "v2", "Message", "body"],
        "label_col_aliases": ["label", "v1", "Category", "class", "target"],
        # values in the label column that mean "this IS scam/spam/fraud"
        "positive_values": ["spam", "scam", "fraud", "1", 1, "smishing"],
        "model_file": os.path.join(MODEL_DIR, "sms_spam_collection.joblib"),
    },
    "url": {
        "path": os.path.join(DATA_DIR, "PhiUSIIL_Phishing_URL_Dataset.csv"),
        "type": "url",
        "text_col_aliases": ["url", "URL", "domain", "link"],
        "label_col_aliases": ["label", "type", "result", "status", "class", "target"],
        "positive_values": [
            "bad", "malicious", "phishing", "scam", "fraud", "defacement",
            "malware", "1", 1,
        ],
        "model_file": os.path.join(MODEL_DIR, "PhiUSIIL_Phishing_URL_Dataset.joblib"),
    },
    "bank": {
        "path": os.path.join(DATA_DIR, "bank-account-fraud.csv"),
        "type": "tabular",
        "label_col_aliases": ["isFraud", "is_fraud", "Class", "fraud", "target", "label"],
        "positive_values": [1, "1", True, "Yes", "fraud"],
        # columns to drop before training even if present (IDs, free text, timestamps as strings)
        "drop_cols": [
            "nameOrig", "nameDest", "isFlaggedFraud", "step", "id",
            "TransactionID", "transaction_id", "date", "Date", "datetime",
        ],
        "model_file": os.path.join(MODEL_DIR, "bank-account-fraud.joblib"),
    },
    "creditcard": {
        "path": os.path.join(DATA_DIR, "creditcard.csv"),
        "type": "tabular",
        "label_col_aliases": ["Class", "class", "isFraud", "fraud", "target", "label"],
        "positive_values": [1, "1", True],
        "drop_cols": ["Time", "id", "TransactionID"],
        "model_file": os.path.join(MODEL_DIR, "creditcard.joblib"),
    },
}

RANDOM_STATE = 42
TEST_SIZE = 0.2