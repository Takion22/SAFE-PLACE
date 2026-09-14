# SAFE-PLACE — conversion des modèles `.pkl` vers `.onnx`

Ce dossier contient un script de conversion **sans modification de SAFE-PLACE**.

## Ce que le script convertit

| Source | ONNX | Entrée |
|---|---|---|
| `models/malagasy_plugin.pkl` | `malagasy_plugin.onnx` | texte |
| `models/malagasy_sentiment_model.pkl` | `malagasy_sentiment_model.onnx` | texte |
| `models/rabbit_hole_model.pkl` | `rabbit_hole_model.onnx` | vecteur numérique |

`malagasy_plugin.pkl` est un dictionnaire qui contient un `TfidfVectorizer`. Le fichier ONNX contient donc le vectorizer, tandis que les métadonnées existantes restent dans le JSON.

## 1. Installer les dépendances

Depuis la racine du projet SAFE-PLACE :

```bash
python -m pip install -r tools/requirements-onnx.txt
```

Le script recommande `scikit-learn==1.9.1`, car les modèles actuels ont été sérialisés avec cette version.

## 2. Convertir + vérifier

```bash
python tools/convert_models_to_onnx.py
```

Le script :

1. charge les `.pkl` existants ;
2. crée les `.onnx` ;
3. vérifie chaque fichier ONNX avec `onnx.checker` ;
4. exécute les deux versions sur les mêmes données de test ;
5. compare les sorties ;
6. crée `models/onnx_manifest.json` ;
7. **ne supprime jamais les `.pkl`**.

## 3. Sortie attendue

```text
models/
├── malagasy_plugin.pkl
├── malagasy_plugin.onnx
├── malagasy_plugin_config.json
├── malagasy_sentiment_model.pkl
├── malagasy_sentiment_model.onnx
├── malagasy_sentiment_config.json
├── rabbit_hole_model.pkl
├── rabbit_hole_model.onnx
├── rabbit_hole_model_config.json
└── onnx_manifest.json
```

## 4. Important

Ne supprime pas encore les `.pkl` et ne modifie pas le Core.

La conversion est une **étape A uniquement**. Après obtention de :

```text
RESULT: CONVERSION SUCCESSFUL
All parity checks passed.
```

on pourra décider séparément de l'étape B : adapter le chargement de SAFE-PLACE pour utiliser ONNX Runtime.

## Option : mettre les ONNX dans un dossier séparé

```bash
python tools/convert_models_to_onnx.py --output-dir models/onnx
```

Cela évite même de mélanger temporairement les deux formats.
