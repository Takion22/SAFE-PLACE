import joblib


def detect_spam(message):
    """
    Charge le modèle entraîné et prédit si un message
    est un spam ou un message normal.
    """

    # Importer le modèle
    model = joblib.load("spam_detector.pkl")

    # Faire la prédiction
    prediction = model.predict([message])[0]

    # Récupérer les probabilités
    probabilities = model.predict_proba([message])[0]

    # Classes utilisées par le modèle
    classes = model.classes_

    # Trouver la probabilité correspondant à la classe prédite
    prediction_index = list(classes).index(prediction)
    confidence = probabilities[prediction_index]

    return prediction, confidence


message = "Congratulations! You have won $1000!"

prediction, confidence = detect_spam(message)

print("Message :", message)
print("Prédiction :", prediction)
print("Confiance :", round(confidence * 100, 2), "%")