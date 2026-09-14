import os
import sys
import pickle
import traceback
import warnings

import numpy as np
import onnx
import onnxruntime as ort

from onnx import helper, TensorProto, numpy_helper

from sklearn import __version__ as sklearn_version
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    StringTensorType,
    FloatTensorType,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "onnx_models"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# UTILS
# ============================================================

def separator():
    print("=" * 70)


def load_pickle(path):
    print(f"Loading: {path}")

    try:
        import joblib

        model = joblib.load(path)

        print(
            f"Loaded successfully with joblib: "
            f"{type(model)}"
        )

        return model

    except Exception as e:
        print(
            f"joblib failed: {e}"
        )

        with open(path, "rb") as f:
            model = pickle.load(f)

        print(
            f"Loaded successfully with pickle: "
            f"{type(model)}"
        )

        return model


def check_onnx(path):
    print()
    print("Checking ONNX structure...")

    model = onnx.load(path)

    onnx.checker.check_model(model)

    print("ONNX structure: OK")

    print()
    print("ONNX INPUTS:")

    for inp in model.graph.input:
        elem_type = (
            inp.type.tensor_type.elem_type
        )

        print(
            f"  - {inp.name} "
            f"type={elem_type}"
        )

    print()
    print("ONNX OUTPUTS:")

    for out in model.graph.output:
        elem_type = (
            out.type.tensor_type.elem_type
        )

        print(
            f"  - {out.name} "
            f"type={elem_type}"
        )


# ============================================================
# MALAGASY PLUGIN
# ============================================================

def convert_malagasy_plugin(model):
    separator()
    print("MALAGASY PLUGIN")
    separator()

    vectorizer = model["vectorizer"]

    print(
        f"Vectorizer: {type(vectorizer)}"
    )

    print(
        f"Vocabulary size: "
        f"{len(vectorizer.vocabulary_)}"
    )

    initial_type = [
        (
            "text",
            StringTensorType([None, 1])
        )
    ]

    print()
    print("Converting TfidfVectorizer...")

    onnx_model = convert_sklearn(
        vectorizer,
        initial_types=initial_type,
        target_opset=17,
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "malagasy_plugin.onnx"
    )

    with open(
        output_path,
        "wb"
    ) as f:
        f.write(
            onnx_model.SerializeToString()
        )

    print(
        f"Saved: {output_path}"
    )

    print(
        f"Size: "
        f"{os.path.getsize(output_path):,} bytes"
    )

    check_onnx(output_path)

    return output_path


def validate_malagasy_plugin(
    model,
    onnx_path
):
    separator()
    print("VALIDATION - MALAGASY PLUGIN")
    separator()

    vectorizer = model["vectorizer"]

    texts = np.array(
        [
            "Salama ry namako",
            "Tena tsara ny andro androany",
            "Tsy tiako mihitsy izany",
            "Misaotra betsaka anao",
        ],
        dtype=object,
    )

    sklearn_output = (
        vectorizer
        .transform(texts)
        .toarray()
        .astype(np.float32)
    )

    session = ort.InferenceSession(
        onnx_path,
        providers=[
            "CPUExecutionProvider"
        ],
    )

    input_name = (
        session.get_inputs()[0].name
    )

    onnx_output = session.run(
        None,
        {
            input_name:
            texts.reshape(-1, 1)
        }
    )[0]

    max_diff = np.max(
        np.abs(
            sklearn_output -
            onnx_output
        )
    )

    print(
        f"sklearn shape: "
        f"{sklearn_output.shape}"
    )

    print(
        f"ONNX shape: "
        f"{onnx_output.shape}"
    )

    print(
        "Maximum absolute difference: "
        f"{max_diff:.10f}"
    )

    passed = (
        sklearn_output.shape ==
        onnx_output.shape
        and
        np.allclose(
            sklearn_output,
            onnx_output,
            rtol=1e-4,
            atol=1e-5,
        )
    )

    print(
        "RESULT: PASS"
        if passed
        else
        "RESULT: FAIL"
    )

    return passed


# ============================================================
# SENTIMENT - LINEAR SVC ONLY
# ============================================================

def convert_sentiment_classifier(
    model
):
    separator()
    print("MALAGASY SENTIMENT")
    separator()

    vectorizer = model.named_steps[
        "tfidf"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    n_features = len(
        vectorizer.vocabulary_
    )

    print(
        f"Vectorizer: {type(vectorizer)}"
    )

    print(
        f"Classifier: {type(classifier)}"
    )

    print(
        f"Analyzer: "
        f"{vectorizer.analyzer}"
    )

    print(
        f"N-gram range: "
        f"{vectorizer.ngram_range}"
    )

    print(
        f"Classes: "
        f"{classifier.classes_}"
    )

    print(
        f"coef shape: "
        f"{classifier.coef_.shape}"
    )

    print(
        f"intercept shape: "
        f"{classifier.intercept_.shape}"
    )

    print(
        f"Number of features: "
        f"{n_features}"
    )

    if (
        classifier.coef_.shape[1]
        != n_features
    ):
        raise RuntimeError(
            "LinearSVC and vectorizer "
            "feature dimensions do not match."
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The sklearn char-level TF-IDF "
        "is NOT converted to ONNX."
    )

    print(
        "The ONNX model therefore receives "
        "the already computed TF-IDF vector."
    )

    print()
    print(
        "Building exact LinearSVC ONNX graph..."
    )

    graph_input = (
        helper.make_tensor_value_info(
            "tfidf_input",
            TensorProto.FLOAT,
            [None, n_features],
        )
    )

    # --------------------------------------------------------
    # COEFFICIENTS
    # --------------------------------------------------------

    coef = np.asarray(
        classifier.coef_,
        dtype=np.float32,
    )

    coef_t = coef.T

    intercept = np.asarray(
        classifier.intercept_,
        dtype=np.float32,
    )

    print()
    print(
        f"coef: "
        f"{coef.shape}"
    )

    print(
        f"coef.T: "
        f"{coef_t.shape}"
    )

    print(
        f"intercept: "
        f"{intercept.shape}"
    )

    coef_initializer = (
        numpy_helper.from_array(
            coef_t,
            name="linear_svc_coef",
        )
    )

    intercept_initializer = (
        numpy_helper.from_array(
            intercept,
            name="linear_svc_intercept",
        )
    )

    classes = np.asarray(
        classifier.classes_,
        dtype=str,
    )

    classes_initializer = helper.make_tensor(
        name="linear_svc_classes",
        data_type=TensorProto.STRING,
        dims=[len(classes)],
        vals=classes.tolist(),
    )

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    nodes = []

    # [N,F] @ [F,1] -> [N,1]
    nodes.append(
        helper.make_node(
            "MatMul",
            [
                "tfidf_input",
                "linear_svc_coef",
            ],
            [
                "linear_svc_matmul_score"
            ],
            name="LinearSVC_MatMul",
        )
    )

    # + intercept
    nodes.append(
        helper.make_node(
            "Add",
            [
                "linear_svc_matmul_score",
                "linear_svc_intercept",
            ],
            [
                "decision_score"
            ],
            name="LinearSVC_Add",
        )
    )

    # [-score, +score]
    nodes.append(
        helper.make_node(
            "Neg",
            [
                "decision_score"
            ],
            [
                "negative_score"
            ],
            name="LinearSVC_Neg",
        )
    )

    nodes.append(
        helper.make_node(
            "Concat",
            [
                "negative_score",
                "decision_score",
            ],
            [
                "class_scores"
            ],
            axis=1,
            name="LinearSVC_ClassScores",
        )
    )

    # class index
    nodes.append(
        helper.make_node(
            "ArgMax",
            [
                "class_scores"
            ],
            [
                "label_index"
            ],
            axis=1,
            keepdims=0,
            name="LinearSVC_ArgMax",
        )
    )

    # class name
    nodes.append(
        helper.make_node(
            "Gather",
            [
                "linear_svc_classes",
                "label_index",
            ],
            [
                "label"
            ],
            axis=0,
            name="LinearSVC_Gather",
        )
    )

    # --------------------------------------------------------
    # OUTPUTS
    # --------------------------------------------------------

    label_output = (
        helper.make_tensor_value_info(
            "label",
            TensorProto.STRING,
            [None],
        )
    )

    decision_output = (
        helper.make_tensor_value_info(
            "decision_score",
            TensorProto.FLOAT,
            [None, 1],
        )
    )

    scores_output = (
        helper.make_tensor_value_info(
            "class_scores",
            TensorProto.FLOAT,
            [None, 2],
        )
    )

    graph = helper.make_graph(
        nodes,
        "MalagasySentimentLinearSVC",
        [
            graph_input
        ],
        [
            label_output,
            decision_output,
            scores_output,
        ],
        [
            coef_initializer,
            intercept_initializer,
            classes_initializer,
        ],
    )

    opset = helper.make_operatorsetid(
        "",
        17
    )

    onnx_model = helper.make_model(
        graph,
        opset_imports=[
            opset
        ],
        producer_name="SAFE-PLACE",
        producer_version="V7",
    )

    onnx.checker.check_model(
        onnx_model
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "malagasy_sentiment_classifier.onnx"
    )

    with open(
        output_path,
        "wb"
    ) as f:
        f.write(
            onnx_model.SerializeToString()
        )

    print()
    print(
        f"Saved: {output_path}"
    )

    print(
        f"Size: "
        f"{os.path.getsize(output_path):,} bytes"
    )

    check_onnx(output_path)

    return output_path


def validate_sentiment_classifier(
    model,
    onnx_path
):
    separator()
    print(
        "VALIDATION - MALAGASY SENTIMENT "
        "CLASSIFIER"
    )
    separator()

    vectorizer = model.named_steps[
        "tfidf"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    # --------------------------------------------------------
    # TEST DATA
    # --------------------------------------------------------

    texts = np.array(
        [
            "Tena tsara ilay vokatra",
            "Tsy tsara mihitsy izany",
            "Mahafinaritra be",
            "Mampalahelo sy ratsy",
            "Tiako ilay zavatra",
            "Tsy tiako izany",
            "Tsara be ny andro",
            "Ratsy ny zavatra nitranga",
            "Faly aho androany",
            "Tena ratsy be",
            "Misaotra betsaka",
            "Tsy misy zavatra tsara",
            "Tena mahafinaritra",
            "Tsy mahafaly ahy",
            "Tsara sy mahafinaritra",
            "Ratsy dia ratsy",
            "Tiako be izany",
            "Tsy tiako mihitsy",
            "Faly sy afa-po aho",
            "Diso fanantenana aho",
        ],
        dtype=object,
    )

    # --------------------------------------------------------
    # SKLEARN TF-IDF
    # --------------------------------------------------------

    sparse_tfidf = (
        vectorizer.transform(texts)
    )

    sklearn_tfidf = (
        sparse_tfidf
        .toarray()
        .astype(np.float32)
    )

    print()
    print(
        "sklearn TF-IDF:"
    )

    print(
        f"  shape = "
        f"{sklearn_tfidf.shape}"
    )

    # --------------------------------------------------------
    # SKLEARN CLASSIFIER
    # --------------------------------------------------------

    sklearn_decision = (
        classifier.decision_function(
            sparse_tfidf
        )
    )

    sklearn_decision = np.asarray(
        sklearn_decision,
        dtype=np.float32,
    )

    if sklearn_decision.ndim == 1:
        sklearn_decision = (
            sklearn_decision.reshape(
                -1,
                1
            )
        )

    sklearn_predictions = (
        classifier.predict(
            sparse_tfidf
        )
    )

    expected_scores = np.concatenate(
        [
            -sklearn_decision,
            sklearn_decision,
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # ONNX
    # --------------------------------------------------------

    session = ort.InferenceSession(
        onnx_path,
        providers=[
            "CPUExecutionProvider"
        ],
    )

    input_name = (
        session.get_inputs()[0].name
    )

    print()
    print(
        "ONNX input:"
    )

    print(
        f"  {input_name}: "
        f"{session.get_inputs()[0].type} "
        f"{session.get_inputs()[0].shape}"
    )

    # --------------------------------------------------------
    # RUN ONNX
    # --------------------------------------------------------

    outputs = session.run(
        None,
        {
            input_name:
            sklearn_tfidf
        }
    )

    output_names = [
        output.name
        for output in session.get_outputs()
    ]

    results = dict(
        zip(
            output_names,
            outputs
        )
    )

    onnx_labels = np.asarray(
        results["label"],
        dtype=str,
    )

    onnx_decision = np.asarray(
        results["decision_score"],
        dtype=np.float32,
    )

    onnx_scores = np.asarray(
        results["class_scores"],
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # TF-IDF SANITY CHECK
    # --------------------------------------------------------

    print()
    print(
        "TF-IDF input:"
    )

    print(
        f"  shape = "
        f"{sklearn_tfidf.shape}"
    )

    print(
        f"  dtype = "
        f"{sklearn_tfidf.dtype}"
    )

    # --------------------------------------------------------
    # DECISION SCORE
    # --------------------------------------------------------

    decision_diff = np.max(
        np.abs(
            sklearn_decision -
            onnx_decision
        )
    )

    print()
    print(
        "Maximum decision score "
        "difference:"
    )

    print(
        f"  {decision_diff:.10f}"
    )

    # --------------------------------------------------------
    # CLASS SCORES
    # --------------------------------------------------------

    score_diff = np.max(
        np.abs(
            expected_scores -
            onnx_scores
        )
    )

    print()
    print(
        "Maximum class score "
        "difference:"
    )

    print(
        f"  {score_diff:.10f}"
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    sklearn_predictions = np.asarray(
        sklearn_predictions,
        dtype=str,
    )

    predictions_match = (
        np.array_equal(
            sklearn_predictions,
            onnx_labels,
        )
    )

    print()
    print(
        "Prediction comparison:"
    )

    for i, (
        sklearn_label,
        onnx_label
    ) in enumerate(
        zip(
            sklearn_predictions,
            onnx_labels,
        )
    ):
        print(
            f"  {i:02d}: "
            f"sklearn={sklearn_label} "
            f"ONNX={onnx_label}"
        )

    print()
    print(
        "Predictions match:",
        predictions_match
    )

    # --------------------------------------------------------
    # NUMERIC VALIDATION
    # --------------------------------------------------------

    decision_match = np.allclose(
        sklearn_decision,
        onnx_decision,
        rtol=1e-5,
        atol=1e-6,
    )

    scores_match = np.allclose(
        expected_scores,
        onnx_scores,
        rtol=1e-5,
        atol=1e-6,
    )

    print(
        "Decision scores match:",
        decision_match
    )

    print(
        "Class scores match:",
        scores_match
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    passed = (
        predictions_match
        and decision_match
        and scores_match
    )

    print()

    if passed:
        print(
            "RESULT: PASS"
        )
    else:
        print(
            "RESULT: FAIL"
        )

    return passed


# ============================================================
# RABBIT HOLE
# ============================================================

def convert_rabbit_hole(model):
    separator()
    print("RABBIT HOLE")
    separator()

    scaler = model.named_steps[
        "scaler"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    n_features = (
        classifier.n_features_in_
    )

    print(
        "Pipeline:"
    )

    print(
        "  scaler:",
        type(scaler).__name__
    )

    print(
        "  classifier:",
        type(classifier).__name__
    )

    print(
        f"Number of features: "
        f"{n_features}"
    )

    initial_type = [
        (
            "float_input",
            FloatTensorType(
                [None, n_features]
            ),
        )
    ]

    print()
    print(
        "Converting Rabbit Hole..."
    )

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=17,
        options={
            id(classifier):
            {
                "zipmap": False
            }
        },
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "rabbit_hole_model.onnx"
    )

    with open(
        output_path,
        "wb"
    ) as f:
        f.write(
            onnx_model.SerializeToString()
        )

    print(
        f"Saved: {output_path}"
    )

    print(
        f"Size: "
        f"{os.path.getsize(output_path):,} bytes"
    )

    check_onnx(output_path)

    return output_path


def validate_rabbit_hole(
    model,
    onnx_path
):
    separator()
    print("VALIDATION - RABBIT HOLE")
    separator()

    classifier = model.named_steps[
        "classifier"
    ]

    n_features = (
        classifier.n_features_in_
    )

    print(
        f"Number of features: "
        f"{n_features}"
    )

    rng = np.random.RandomState(
        42
    )

    X = rng.normal(
        size=(20, n_features)
    ).astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )

        sklearn_predictions = (
            model.predict(X)
        )

        sklearn_probabilities = (
            model.predict_proba(X)
        )

    session = ort.InferenceSession(
        onnx_path,
        providers=[
            "CPUExecutionProvider"
        ],
    )

    input_name = (
        session.get_inputs()[0].name
    )

    outputs = session.run(
        None,
        {
            input_name: X
        }
    )

    output_names = [
        output.name
        for output in session.get_outputs()
    ]

    results = dict(
        zip(
            output_names,
            outputs
        )
    )

    onnx_predictions = np.asarray(
        results["label"]
    )

    onnx_probabilities = np.asarray(
        results["probabilities"]
    )

    print()
    print(
        "Outputs:"
    )

    print(
        f"  label: "
        f"shape={onnx_predictions.shape}"
    )

    print(
        f"  probabilities: "
        f"shape={onnx_probabilities.shape}"
    )

    print()
    print(
        "Prediction comparison:"
    )

    for i, (
        sklearn_label,
        onnx_label
    ) in enumerate(
        zip(
            sklearn_predictions,
            onnx_predictions,
        )
    ):
        print(
            f"  {i:02d}: "
            f"sklearn={sklearn_label} "
            f"ONNX={onnx_label}"
        )

    predictions_match = (
        np.array_equal(
            sklearn_predictions,
            onnx_predictions,
        )
    )

    probability_diff = np.max(
        np.abs(
            sklearn_probabilities -
            onnx_probabilities
        )
    )

    probabilities_match = (
        np.allclose(
            sklearn_probabilities,
            onnx_probabilities,
            rtol=1e-4,
            atol=1e-5,
        )
    )

    print()
    print(
        "Maximum probability difference:",
        f"{probability_diff:.10f}"
    )

    print(
        "Predictions match:",
        predictions_match
    )

    print(
        "Probabilities match:",
        probabilities_match
    )

    passed = (
        predictions_match
        and probabilities_match
    )

    print()

    print(
        "RESULT: PASS"
        if passed
        else
        "RESULT: FAIL"
    )

    return passed


# ============================================================
# MAIN
# ============================================================

def convert_all():

    separator()
    print(
        "SAFE-PLACE ONNX CONVERTER V7"
    )
    separator()

    print(
        f"Python: {sys.version}"
    )

    print(
        f"sklearn: "
        f"{sklearn_version}"
    )

    print(
        f"onnx: "
        f"{onnx.__version__}"
    )

    print(
        f"onnxruntime: "
        f"{ort.__version__}"
    )

    print()
    print(
        "Base directory:"
    )
    print(BASE_DIR)

    print()
    print(
        "Models directory:"
    )
    print(MODELS_DIR)

    print()
    print(
        "Output directory:"
    )
    print(OUTPUT_DIR)

    results = {}

    # ========================================================
    # PLUGIN
    # ========================================================

    try:

        path = os.path.join(
            MODELS_DIR,
            "malagasy_plugin.pkl"
        )

        model = load_pickle(path)

        onnx_path = (
            convert_malagasy_plugin(
                model
            )
        )

        results[
            "malagasy_plugin"
        ] = (
            validate_malagasy_plugin(
                model,
                onnx_path
            )
        )

    except Exception:

        results[
            "malagasy_plugin"
        ] = False

        separator()

        print(
            "ERROR - MALAGASY PLUGIN"
        )

        separator()

        print(
            traceback.format_exc()
        )

    # ========================================================
    # SENTIMENT
    # ========================================================

    try:

        path = os.path.join(
            MODELS_DIR,
            "malagasy_sentiment_model.pkl"
        )

        model = load_pickle(path)

        onnx_path = (
            convert_sentiment_classifier(
                model
            )
        )

        results[
            "malagasy_sentiment_classifier"
        ] = (
            validate_sentiment_classifier(
                model,
                onnx_path
            )
        )

    except Exception:

        results[
            "malagasy_sentiment_classifier"
        ] = False

        separator()

        print(
            "ERROR - MALAGASY SENTIMENT"
        )

        separator()

        print(
            traceback.format_exc()
        )

    # ========================================================
    # RABBIT HOLE
    # ========================================================

    try:

        path = os.path.join(
            MODELS_DIR,
            "rabbit_hole_model.pkl"
        )

        model = load_pickle(path)

        onnx_path = (
            convert_rabbit_hole(
                model
            )
        )

        results[
            "rabbit_hole_model"
        ] = (
            validate_rabbit_hole(
                model,
                onnx_path
            )
        )

    except Exception:

        results[
            "rabbit_hole_model"
        ] = False

        separator()

        print(
            "ERROR - RABBIT HOLE"
        )

        separator()

        print(
            traceback.format_exc()
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    separator()

    print(
        "FINAL SUMMARY"
    )

    separator()

    print()
    print(
        "VALIDATION:"
    )

    for name, passed in results.items():

        print(
            f"  {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print()

    all_passed = all(
        results.values()
    )

    if all_passed:

        separator()

        print(
            "STATUS: ALL CONVERTED "
            "COMPONENTS PASSED"
        )

        separator()

    else:

        separator()

        print(
            "STATUS: SOME COMPONENTS FAILED"
        )

        separator()

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Les fichiers .pkl originaux "
        "n'ont PAS été modifiés."
    )

    print()
    print(
        "Les fichiers ONNX sont dans:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    convert_all()