from abc import ABC, abstractmethod


class SafePlaceComponent(ABC):

    @abstractmethod
    def name(self):
        """Nom du composant."""
        raise NotImplementedError

    @abstractmethod
    def analyze(self, data):
        """Analyse générique."""
        raise NotImplementedError


class SafePlaceTextComponent(SafePlaceComponent):

    @abstractmethod
    def analyze(self, text, language=None, context=None):
        """Analyse d'un texte."""
        raise NotImplementedError


class SafePlaceBehaviorComponent(SafePlaceComponent):

    @abstractmethod
    def analyze(self, data):
        """Analyse de données comportementales."""
        raise NotImplementedError