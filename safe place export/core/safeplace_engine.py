from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parent.parent

# Permet d'importer les modules depuis la racine du projet.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from modules.RABBIT_HOLE import RabbitHoleModel
from core.language_router import LanguageRouter


class SafePlaceEngine:

    def __init__(self):

        # -------------------------------------------------
        # Rabbit Hole
        # -------------------------------------------------

        self.rabbit_hole = RabbitHoleModel()

        # -------------------------------------------------
        # Language Router
        # -------------------------------------------------

        self.language_router = LanguageRouter()

    # =====================================================
    # LANGUAGE
    # =====================================================

    def detect_language(self, text):

        return self.language_router.detect(text)

    def route_text(self, text, language=None):

        return self.language_router.route(
            text,
            language=language
        )

    # =====================================================
    # RABBIT HOLE
    # =====================================================

    def analyze_rabbit_hole(self, metrics):

        return self.rabbit_hole.predict(
            metrics
        )

    # =====================================================
    # STATUS
    # =====================================================

    def status(self):

        return {
            "status": "ok",
            "components": {
                "language_router": True,
                "rabbit_hole": True
            }
        }


def main():

    engine = SafePlaceEngine()

    print("=" * 70)
    print("SAFEPLACE ENGINE")
    print("=" * 70)

    print("\nSTATUS")
    print(engine.status())

    # -------------------------------------------------
    # Language routing
    # -------------------------------------------------

    tests = [
        "Cette vidéo est intéressante.",
        "This video is interesting.",
        "Mahafinaritra be ity horonantsary ity."
    ]

    for text in tests:

        print("\nTEXT")
        print(text)

        print(
            engine.route_text(text)
        )

    # -------------------------------------------------
    # Rabbit Hole
    # -------------------------------------------------

    print("\nRABBIT HOLE")

    metrics = {
        "sequence_length": 120,
        "click_depth": 120,
        "session_time": 5000,
        "unique_videos": 90,
        "repetition_score": 0.30,
        "diversity_score": 0.70
    }

    print(
        engine.analyze_rabbit_hole(
            metrics
        )
    )


if __name__ == "__main__":
    main()