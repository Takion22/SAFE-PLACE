from pathlib import Path
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_FILE = BASE_DIR / "models" / "rabbit_hole_model.pkl"


FEATURES = [
    "sequence_length",
    "click_depth",
    "session_time",
    "unique_videos",
    "repetition_score",
    "diversity_score",
]


class RabbitHoleModel:

    def __init__(self):
        if not MODEL_FILE.exists():
            raise FileNotFoundError(
                f"Modèle Rabbit Hole introuvable : {MODEL_FILE}"
            )

        self.model = joblib.load(MODEL_FILE)

    def predict(self, metrics):

        missing = [
            feature
            for feature in FEATURES
            if feature not in metrics
        ]

        if missing:
            raise ValueError(
                f"Variables manquantes : {missing}"
            )

        X = pd.DataFrame(
            [[metrics[feature] for feature in FEATURES]],
            columns=FEATURES
        )

        prediction = self.model.predict(X)[0]

        result = {
            "prediction": str(prediction)
        }

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)[0]

            result["probabilities"] = {
                str(label): float(probability)
                for label, probability in zip(
                    self.model.classes_,
                    probabilities
                )
            }

        return result