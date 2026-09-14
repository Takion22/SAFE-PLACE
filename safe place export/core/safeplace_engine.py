from pathlib import Path
import sys
import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

CORE_DIR = Path(__file__).resolve().parent

if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from language_router import LanguageRouter


class SafePlaceEngine:

    def __init__(self):

        self.models_dir = MODELS_DIR

        # -------------------------------------------------
        # Rabbit Hole
        # -------------------------------------------------

        rabbit_hole_file = (
            self.models_dir
            / "rabbit_hole_model.pkl"
        )

        if not rabbit_hole_file.exists():
            raise FileNotFoundError(
                f"Rabbit Hole model introuvable : "
                f"{rabbit_hole_file}"
            )

        self.rabbit_hole_model = joblib.load(
            rabbit_hole_file
        )

        # -------------------------------------------------
        # Malagasy Plugin
        # -------------------------------------------------

        malagasy_file = (
            self.models_dir
            / "malagasy_plugin.pkl"
        )

        self.malagasy_plugin = None

        if malagasy_file.exists():

            self.malagasy_plugin = joblib.load(
                malagasy_file
            )

        # -------------------------------------------------
        # Language Router
        # -------------------------------------------------

        self.language_router = LanguageRouter(
            malagasy_plugin=self.malagasy_plugin
        )

    # =====================================================
    # LANGUAGE
    # =====================================================

    def detect_language(self, text):

        return self.language_router.detect(text)

    # =====================================================
    # TEXT ROUTING
    # =====================================================

    def route_text(self, text, language=None):

        route = self.language_router.route(
            text,
            language=language
        )

        result = {
            "text": text,
            "language": route["language"],
            "target": route["target"]
        }

        # -------------------------------------------------
        # Malagasy
        # -------------------------------------------------

        if (
            route["language"] == "mg"
            and self.malagasy_plugin is not None
        ):

            vectorizer = self.malagasy_plugin["vectorizer"]

            vector = vectorizer.transform(
                [text]
            )

            result["malagasy"] = {
                "vector_shape": list(
                    vector.shape
                ),
                "active_features": int(
                    vector.nnz
                ),
                "vocabulary_size": int(
                    self.malagasy_plugin.get(
                        "vocabulary_size",
                        0
                    )
                )
            }

        return result

    # =====================================================
    # RABBIT HOLE
    # =====================================================

    def analyze_rabbit_hole(self, metrics):

        required_features = [
            "sequence_length",
            "click_depth",
            "session_time",
            "unique_videos",
            "repetition_score",
            "diversity_score"
        ]

        missing = [
            feature
            for feature in required_features
            if feature not in metrics
        ]

        if missing:
            raise ValueError(
                f"Variables manquantes : {missing}"
            )

        import pandas as pd

        X = pd.DataFrame(
            [[
                metrics["sequence_length"],
                metrics["click_depth"],
                metrics["session_time"],
                metrics["unique_videos"],
                metrics["repetition_score"],
                metrics["diversity_score"]
            ]],
            columns=required_features
        )

        prediction = self.rabbit_hole_model.predict(X)[0]

        result = {
            "prediction": str(prediction)
        }

        if hasattr(
            self.rabbit_hole_model,
            "predict_proba"
        ):

            probabilities = (
                self.rabbit_hole_model
                .predict_proba(X)[0]
            )

            result["probabilities"] = {
                str(label): float(probability)
                for label, probability in zip(
                    self.rabbit_hole_model.classes_,
                    probabilities
                )
            }

        return result

    # =====================================================
    # HEALTH CHECK
    # =====================================================

    def status(self):

        return {
            "status": "ok",
            "components": {
                "rabbit_hole": (
                    self.rabbit_hole_model
                    is not None
                ),
                "malagasy_plugin": (
                    self.malagasy_plugin
                    is not None
                ),
                "language_router": (
                    self.language_router
                    is not None
                )
            }
        }