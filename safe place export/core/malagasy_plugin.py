"""
SAFE-PLACE Malagasy Language Plugin

Standalone Malagasy capability for SAFE-PLACE.

Design rules:
- Does not modify or import the SAFE-PLACE engine.
- Does not modify FR/EN models.
- Does not translate input.
- Can be called independently by the router/core.
- Returns a stable, generic result while preserving the legacy result fields.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import json

import joblib


class MalagasyPlugin:
    """Standalone Malagasy analysis provider."""

    LANGUAGE = "mg"
    LANGUAGE_NAME = "Malagasy"
    PLUGIN_TYPE = "language_plugin"
    VERSION = "1.1"

    def __init__(
        self,
        plugin_file: Optional[str] = None,
        sentiment_file: Optional[str] = None,
        config_file: Optional[str] = None,
        sentiment_config_file: Optional[str] = None,
        load_sentiment: bool = True,
    ):
        base_dir = Path(__file__).resolve().parent.parent
        models_dir = base_dir / "models"

        self.plugin_file = Path(
            plugin_file or models_dir / "malagasy_plugin.pkl"
        )
        self.sentiment_file = Path(
            sentiment_file or models_dir / "malagasy_sentiment_model.pkl"
        )
        self.config_file = Path(
            config_file or models_dir / "malagasy_plugin_config.json"
        )
        self.sentiment_config_file = Path(
            sentiment_config_file
            or models_dir / "malagasy_sentiment_config.json"
        )

        self.config = self._load_json(self.config_file)
        self.sentiment_config = self._load_json(
            self.sentiment_config_file
        )

        self.vectorizer = None
        self.sentiment_model = None

        self.language = self.config.get(
            "language",
            self.LANGUAGE_NAME,
        )
        self.vocabulary_size = self.config.get(
            "vocabulary_size",
            0,
        )

        self._load_plugin_model()

        if load_sentiment and self.sentiment_file.exists():
            self._load_sentiment_model()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}

        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return {}

        return data if isinstance(data, dict) else {}

    def _load_plugin_model(self) -> None:
        if not self.plugin_file.exists():
            raise FileNotFoundError(
                "Plugin Malagasy introuvable : "
                + str(self.plugin_file)
            )

        data = joblib.load(self.plugin_file)

        if not isinstance(data, dict):
            raise TypeError(
                "malagasy_plugin.pkl doit contenir un dictionnaire."
            )

        vectorizer = data.get("vectorizer")
        if vectorizer is None or not hasattr(vectorizer, "transform"):
            raise TypeError(
                "Le plugin Malagasy doit fournir un vectorizer "
                "compatible avec transform()."
            )

        self.vectorizer = vectorizer
        self.language = data.get("language", self.language)
        self.vocabulary_size = int(
            data.get("vocabulary_size", self.vocabulary_size or 0)
        )

    def _load_sentiment_model(self) -> None:
        try:
            self.sentiment_model = joblib.load(self.sentiment_file)
        except Exception:
            # Sentiment is an optional capability. The language plugin
            # itself remains usable if this optional model cannot load.
            self.sentiment_model = None

    # ------------------------------------------------------------------
    # Generic plugin interface
    # ------------------------------------------------------------------

    def name(self) -> str:
        return "malagasy_plugin"

    def supports(self, language: Optional[str]) -> bool:
        """Return True only when the requested language includes Malagasy."""
        if language is None:
            return False

        normalized = str(language).strip().lower()

        if normalized in {"mg", "malagasy"}:
            return True

        # Accept common router representations such as:
        # "fr+mg", "mg+fr", "en,m g" (after whitespace removal), etc.
        compact = normalized.replace(" ", "")
        parts = {
            part
            for part in compact.replace(",", "+").replace("/", "+").split("+")
            if part
        }

        return "mg" in parts or "malagasy" in parts

    def capabilities(self) -> Dict[str, Any]:
        """Describe available capabilities without exposing implementation."""
        return {
            "plugin": self.name(),
            "type": self.PLUGIN_TYPE,
            "language": self.LANGUAGE,
            "language_name": self.language,
            "version": self.VERSION,
            "capabilities": {
                "language_features": self.vectorizer is not None,
                "sentiment": self.sentiment_model is not None,
            },
            "vocabulary_size": self.vocabulary_size,
        }

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def transform(self, text: str):
        if not isinstance(text, str):
            raise TypeError(
                "text doit être une chaîne de caractères."
            )

        if not text.strip():
            raise ValueError("text ne peut pas être vide.")

        return self.vectorizer.transform([text])

    def transform_many(self, texts: Sequence[str]):
        if not isinstance(texts, (list, tuple)):
            raise TypeError(
                "texts doit être une liste ou un tuple de chaînes."
            )

        if not all(isinstance(text, str) for text in texts):
            raise TypeError(
                "Tous les éléments de texts doivent être des chaînes."
            )

        return self.vectorizer.transform(texts)

    # ------------------------------------------------------------------
    # Sentiment
    # ------------------------------------------------------------------

    def sentiment(self, text: str):
        """Return the model's sentiment label, or None if unavailable."""
        if self.sentiment_model is None:
            return None

        prediction = self.sentiment_model.predict([text])
        if len(prediction) == 0:
            return None

        return str(prediction[0])

    # ------------------------------------------------------------------
    # SAFE-PLACE-facing result
    # ------------------------------------------------------------------

    def analyze(
        self,
        text: str,
        language: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze Malagasy input without translating or altering the text.

        `language` is optional so the plugin can be used directly or after
        LanguageRouter detection. `context` is accepted for compatibility
        with generic SAFE-PLACE text-component interfaces but is not required.
        """
        if not isinstance(text, str):
            raise TypeError(
                "text doit être une chaîne de caractères."
            )

        if not text.strip():
            raise ValueError("text ne peut pas être vide.")

        vector = self.transform(text)

        result = {
            # Generic contract
            "plugin": self.name(),
            "language": self.LANGUAGE,
            "language_name": self.language,
            "active": True,
            "input_text": text,
            "language_confidence": None,

            # Backward-compatible fields from the previous plugin
            "vector_shape": list(vector.shape),
            "active_features": int(vector.nnz),
            "vocabulary_size": self.vocabulary_size,
        }

        if self.sentiment_model is not None:
            result["sentiment"] = self.sentiment(text)

        # Optional metadata supplied by the caller is kept separate.
        if context is not None:
            result["context_available"] = True

        if language is not None:
            result["routed_language"] = language

        return result

    # `process` is the generic entry point for future SAFE-PLACE modules.
    def process(
        self,
        text: str,
        language: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.analyze(
            text,
            language=language,
            context=context,
        )
