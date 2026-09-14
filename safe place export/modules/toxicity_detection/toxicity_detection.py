import joblib


# Charger le modèle entraîné
model = joblib.load("toxic_detector.pkl")


def detect_toxicity(message):
    """
    Détecte si un message anglais est toxique ou non.

    Retourne :
        result      : TOXIQUE ou NON TOXIQUE
        probability : probabilité que le message soit toxique
    """

    # Prédiction
    prediction = model.predict([message])[0]

    # Probabilités
    probabilities = model.predict_proba([message])[0]

    # Récupérer la position de la classe 1 (toxique)
    toxic_index = list(model.classes_).index(1)

    toxic_probability = probabilities[toxic_index]

    # Résultat
    if prediction == 1:
        result = "TOXIQUE"
    else:
        result = "NON TOXIQUE"

    return result, toxic_probability


# ============================================================
# EXEMPLE
# ============================================================

message = "You are stupid and nobody likes you."

result, probability = detect_toxicity(message)

print("Message :", message)
print("Résultat :", result)
print("Probabilité toxique :", round(probability * 100, 2), "%")