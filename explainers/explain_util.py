"""
XAI utilities for the scam/fraud detection backend.

LIME is used for local feature explanations.
Anchor is loaded optionally so that a missing Anchor installation
does NOT prevent the FastAPI application from starting.
"""

import numpy as np

from lime.lime_text import LimeTextExplainer
from lime.lime_tabular import LimeTabularExplainer


# ---------------------------------------------------------
# OPTIONAL ANCHOR IMPORT
# ---------------------------------------------------------

try:
    from anchor.anchor_text import AnchorText
    from anchor.anchor_tabular import AnchorTabularExplainer

    ANCHOR_AVAILABLE = True

except ImportError:
    AnchorText = None
    AnchorTabularExplainer = None
    ANCHOR_AVAILABLE = False


# =========================================================
# LIME - TEXT
# =========================================================

def lime_explain_text(
    model,
    class_names,
    text: str,
    num_features: int = 10
):
    """
    Explain a text prediction using LIME.

    model:
        sklearn Pipeline with predict_proba()

    class_names:
        Example: ["legit", "scam"]

    text:
        Raw SMS/message/description.

    Returns:
        List of (word, weight).
    """

    text = str(text)

    explainer = LimeTextExplainer(
        class_names=list(class_names)
    )

    def predict_fn(texts):
        texts = [str(x) for x in texts]
        return np.asarray(
            model.predict_proba(texts)
        )

    explanation = explainer.explain_instance(
        text,
        predict_fn,
        num_features=num_features
    )

    return explanation.as_list()


# =========================================================
# ANCHOR - TEXT
# =========================================================

def anchor_explain_text(
    model,
    class_names,
    text: str,
    threshold: float = 0.90
):
    """
    Explain a text prediction using Anchor.

    If Anchor is not installed, returns a clear status
    instead of crashing the backend.
    """

    if not ANCHOR_AVAILABLE:
        return {
            "anchor": [],
            "precision": None,
            "coverage": None,
            "available": False,
            "message": "Anchor is not installed."
        }

    text = str(text)

    def predict_fn(texts):
        texts = [str(x) for x in texts]

        probabilities = np.asarray(
            model.predict_proba(texts)
        )

        return np.argmax(
            probabilities,
            axis=1
        )

    try:
        explainer = AnchorText(
            class_names=list(class_names),
            use_unk_distribution=True
        )

        explanation = explainer.explain_instance(
            text,
            predict_fn,
            threshold=threshold
        )

        return {
            "anchor": explanation.names(),
            "precision": float(explanation.precision()),
            "coverage": float(explanation.coverage()),
            "available": True
        }

    except Exception as exc:
        return {
            "anchor": [],
            "precision": None,
            "coverage": None,
            "available": True,
            "message": str(exc)
        }


# =========================================================
# LIME - TABULAR
# =========================================================

def lime_explain_tabular(
    model,
    class_names,
    x_row,
    feature_names,
    training_data,
    categorical_features=None,
    num_features=10
):
    """
    Explain one tabular prediction using LIME.
    """

    training_data = np.asarray(training_data)
    x_row = np.asarray(x_row)

    explainer = LimeTabularExplainer(
        training_data=training_data,
        feature_names=list(feature_names),
        class_names=list(class_names),
        categorical_features=categorical_features,
        mode="classification"
    )

    def predict_fn(rows):
        rows = np.asarray(rows)

        return np.asarray(
            model.predict_proba(rows)
        )

    explanation = explainer.explain_instance(
        x_row,
        predict_fn,
        num_features=num_features
    )

    return explanation.as_list()


# =========================================================
# ANCHOR - TABULAR
# =========================================================

def anchor_explain_tabular(
    model,
    class_names,
    x_row,
    feature_names,
    training_data,
    categorical_names=None,
    threshold=0.90
):
    """
    Explain one tabular prediction using Anchor.

    Anchor is optional. If unavailable, the backend remains
    operational and returns a status object.
    """

    if not ANCHOR_AVAILABLE:
        return {
            "anchor": [],
            "precision": None,
            "coverage": None,
            "available": False,
            "message": "Anchor is not installed."
        }

    training_data = np.asarray(training_data)
    x_row = np.asarray(x_row)

    try:
        explainer = AnchorTabularExplainer(
            class_names=list(class_names),
            feature_names=list(feature_names),
            train_data=training_data,
            categorical_names=categorical_names or {}
        )

        def predict_fn(rows):
            rows = np.asarray(rows)

            return np.asarray(
                model.predict(rows)
            )

        explanation = explainer.explain_instance(
            x_row,
            predict_fn,
            threshold=threshold
        )

        return {
            "anchor": explanation.names(),
            "precision": float(explanation.precision()),
            "coverage": float(explanation.coverage()),
            "available": True
        }

    except Exception as exc:
        return {
            "anchor": [],
            "precision": None,
            "coverage": None,
            "available": True,
            "message": str(exc)
        }