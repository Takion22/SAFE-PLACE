import re


class LanguageRouter:

    LANGUAGE_FR = "fr"
    LANGUAGE_EN = "en"
    LANGUAGE_MG = "mg"
    LANGUAGE_UNKNOWN = "unknown"

    FR_MARKERS = {
        "le", "la", "les", "un", "une", "des",
        "de", "du", "et", "est", "sont",
        "dans", "pour", "avec", "ce", "cette",
        "ces", "que", "qui", "sur", "pas",
        "mais", "comme", "plus", "une"
    }

    EN_MARKERS = {
        "the", "a", "an", "and", "is", "are",
        "was", "were", "of", "to", "for",
        "with", "this", "that", "in", "on",
        "not", "but", "more", "very"
    }

    MG_MARKERS = {
        "ny", "dia", "izy", "ity", "ireo",
        "amin", "amin'ny", "ho", "fa", "tsy",
        "ary", "no", "ka", "ao", "be",
        "izany", "mety", "nataony",
        "mahafinaritra", "horonantsary",
        "governemanta"
    }

    def __init__(self, malagasy_plugin=None):
        self.malagasy_plugin = malagasy_plugin

    @staticmethod
    def _tokens(text):
        return re.findall(
            r"\b[\wÀ-ÿ'-]+\b",
            text.lower()
        )

    def detect(self, text):

        if not isinstance(text, str):
            raise TypeError(
                "text doit être une chaîne de caractères."
            )

        tokens = self._tokens(text)

        if not tokens:
            return self.LANGUAGE_UNKNOWN

        token_set = set(tokens)

        fr_score = sum(
            token in self.FR_MARKERS
            for token in token_set
        )

        en_score = sum(
            token in self.EN_MARKERS
            for token in token_set
        )

        mg_score = sum(
            token in self.MG_MARKERS
            for token in token_set
        )

        scores = {
            self.LANGUAGE_FR: fr_score,
            self.LANGUAGE_EN: en_score,
            self.LANGUAGE_MG: mg_score
        }

        best_language = max(
            scores,
            key=scores.get
        )

        best_score = scores[best_language]

        if best_score == 0:
            return self.LANGUAGE_UNKNOWN

        return best_language

    def route(self, text, language=None):

        detected_language = (
            language
            if language is not None
            else self.detect(text)
        )

        if detected_language == self.LANGUAGE_MG:
            target = "malagasy_plugin"

        elif detected_language in (
            self.LANGUAGE_FR,
            self.LANGUAGE_EN
        ):
            target = "native"

        else:
            target = "unknown"

        return {
            "language": detected_language,
            "target": target
        }