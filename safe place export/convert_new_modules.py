import os
import warnings
import joblib
import numpy as np

import onnx
import onnxruntime as ort

from onnx import helper, TensorProto, numpy_helper


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PKL_PATH = os.path.join(
    BASE_DIR,
    "modules",
    "toxicity_detection",
    "toxic_detector.pkl"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "onnx_models"
)

ONNX_PATH = os.path.join(
    OUTPUT_DIR,
    "toxic_detector_classifier.onnx"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

def load_model():

    print("=" * 70)
    print("LOADING TOXIC DETECTOR")
    print("=" * 70)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(PKL_PATH)

    print("Type:", type(model).__name__)

    for name, component in model.steps:
        print(
            " ",
            name,
            "=>",
            type(component).__name__
        )

    return model


# ============================================================
# INSPECTION
# ============================================================

def inspect_model(model):

    tfidf = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    print()
    print("-" * 70)
    print("MODEL INSPECTION")
    print("-" * 70)

    print("Features:", len(tfidf.vocabulary_))
    print("Analyzer:", tfidf.analyzer)
    print("N-grams:", tfidf.ngram_range)
    print("Stop words:", tfidf.stop_words)

    print()
    print("Classes:", classifier.classes_)
    print("Coefficient shape:", classifier.coef_.shape)
    print("Intercept:", classifier.intercept_)

    print()
    print("Coefficient dtype:", classifier.coef_.dtype)
    print("Intercept dtype:", classifier.intercept_.dtype)


# ============================================================
# CREATE ONNX LOGISTIC REGRESSION
# ============================================================

def create_onnx_classifier(model):

    classifier = model.named_steps["classifier"]

    coef = np.asarray(
        classifier.coef_,
        dtype=np.float64
    )

    intercept = np.asarray(
        classifier.intercept_,
        dtype=np.float64
    )

    classes = np.asarray(
        classifier.classes_
    )

    n_features = coef.shape[1]

    print()
    print("-" * 70)
    print("BUILDING ONNX CLASSIFIER")
    print("-" * 70)

    print("Input features:", n_features)
    print("Coefficient shape:", coef.shape)
    print("Using dtype: float64")

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    input_tensor = helper.make_tensor_value_info(
        "tfidf_input",
        TensorProto.DOUBLE,
        [None, n_features]
    )

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------

    label_tensor = helper.make_tensor_value_info(
        "label",
        TensorProto.INT64,
        [None]
    )

    probability_tensor = helper.make_tensor_value_info(
        "probabilities",
        TensorProto.DOUBLE,
        [None, 2]
    )

    decision_tensor = helper.make_tensor_value_info(
        "decision_score",
        TensorProto.DOUBLE,
        [None]
    )

    # --------------------------------------------------------
    # Constants
    # --------------------------------------------------------

    coef_tensor = numpy_helper.from_array(
        coef.T,
        name="coef"
    )

    intercept_tensor = numpy_helper.from_array(
        intercept,
        name="intercept"
    )

    class_tensor = numpy_helper.from_array(
        classes.astype(np.int64),
        name="classes"
    )

    zero_tensor = numpy_helper.from_array(
        np.array([0.0], dtype=np.float64),
        name="zero"
    )

    # --------------------------------------------------------
    # Linear decision function
    #
    # X @ coef.T + intercept
    # --------------------------------------------------------

    matmul_node = helper.make_node(
        "MatMul",
        ["tfidf_input", "coef"],
        ["linear_score"],
        name="LogisticRegression_MatMul"
    )

    add_node = helper.make_node(
        "Add",
        ["linear_score", "intercept"],
        ["decision_score"],
        name="LogisticRegression_Add"
    )

    # --------------------------------------------------------
    # sigmoid(score)
    #
    # p(class 1)
    # --------------------------------------------------------

    sigmoid_node = helper.make_node(
        "Sigmoid",
        ["decision_score"],
        ["positive_probability"],
        name="LogisticRegression_Sigmoid"
    )

    # --------------------------------------------------------
    # p(class 0) = 1 - p(class 1)
    # --------------------------------------------------------

    negative_probability_node = helper.make_node(
        "Sub",
        ["one", "positive_probability"],
        ["negative_probability"],
        name="NegativeProbability"
    )

    # --------------------------------------------------------
    # Build "one" dynamically
    # --------------------------------------------------------

    one_tensor = numpy_helper.from_array(
        np.array([1.0], dtype=np.float64),
        name="one"
    )

    # --------------------------------------------------------
    # probability matrix
    #
    # [p(class0), p(class1)]
    # --------------------------------------------------------

    concat_node = helper.make_node(
        "Concat",
        [
            "negative_probability",
            "positive_probability"
        ],
        ["probabilities"],
        axis=1,
        name="ProbabilityConcat"
    )

    # --------------------------------------------------------
    # class index
    #
    # argmax(probabilities)
    # --------------------------------------------------------

    argmax_node = helper.make_node(
        "ArgMax",
        ["probabilities"],
        ["class_index"],
        axis=1,
        keepdims=0,
        name="ProbabilityArgMax"
    )

    # --------------------------------------------------------
    # class label
    # --------------------------------------------------------

    gather_node = helper.make_node(
        "Gather",
        ["classes", "class_index"],
        ["label"],
        axis=0,
        name="ClassGather"
    )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    graph = helper.make_graph(
        [
            matmul_node,
            add_node,
            sigmoid_node,
            negative_probability_node,
            concat_node,
            argmax_node,
            gather_node
        ],
        "ToxicDetector_LogisticRegression",
        [input_tensor],
        [
            label_tensor,
            probability_tensor,
            decision_tensor
        ],
        initializer=[
            coef_tensor,
            intercept_tensor,
            class_tensor,
            one_tensor
        ]
    )

    model_onnx = helper.make_model(
        graph,
        producer_name="SAFE-PLACE",
        opset_imports=[
            helper.make_operatorsetid("", 17)
        ]
    )

    onnx.checker.check_model(model_onnx)

    with open(ONNX_PATH, "wb") as f:
        f.write(
            model_onnx.SerializeToString()
        )

    print()
    print("ONNX created:")
    print(ONNX_PATH)

    print(
        "Size:",
        os.path.getsize(ONNX_PATH),
        "bytes"
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(model):

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Test data
    # --------------------------------------------------------

    tests = [
        "hello",
        "hello friend",
        "you won a prize",
        "click this link now",
        "this is a normal message",
        "you are stupid",
        "I hate you",
        "have a great day",
        "free money available",
        "thank you very much",
        "you idiot",
        "please contact me",
        "",
        "this is completely harmless",
        "you are a terrible person",
        "CONGRATULATIONS YOU WON",
        "meeting at 10 tomorrow",
        "buy now limited offer",
        "I really appreciate your help",
        "go away",
    ]

    # --------------------------------------------------------
    # sklearn TF-IDF
    # --------------------------------------------------------

    tfidf = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    X = tfidf.transform(tests)

    # sklearn's vectorizer normally produces float64
    X_dense = X.toarray().astype(
        np.float64
    )

    print()
    print("TF-IDF shape:", X_dense.shape)
    print("TF-IDF dtype:", X_dense.dtype)

    # --------------------------------------------------------
    # sklearn reference
    # --------------------------------------------------------

    sklearn_decision = classifier.decision_function(
        X_dense
    )

    sklearn_probability = classifier.predict_proba(
        X_dense
    )

    sklearn_prediction = classifier.predict(
        X_dense
    )

    # --------------------------------------------------------
    # ONNX
    # --------------------------------------------------------

    session = ort.InferenceSession(
        ONNX_PATH,
        providers=["CPUExecutionProvider"]
    )

    print()
    print("ONNX input:")
    print(
        session.get_inputs()[0].name,
        session.get_inputs()[0].type,
        session.get_inputs()[0].shape
    )

    outputs = session.run(
        None,
        {
            "tfidf_input": X_dense
        }
    )

    # Output order:
    # 0 = label
    # 1 = probabilities
    # 2 = decision_score

    onnx_prediction = np.asarray(
        outputs[0]
    ).reshape(-1)

    onnx_probability = np.asarray(
        outputs[1]
    )

    onnx_decision = np.asarray(
        outputs[2]
    ).reshape(-1)

    # --------------------------------------------------------
    # Decision score
    # --------------------------------------------------------

    decision_diff = np.max(
        np.abs(
            sklearn_decision -
            onnx_decision
        )
    )

    print()
    print(
        "Max decision score difference:",
        f"{decision_diff:.15f}"
    )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probability_diff = np.max(
        np.abs(
            sklearn_probability -
            onnx_probability
        )
    )

    print(
        "Max probability difference:",
        f"{probability_diff:.15f}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    prediction_match = np.array_equal(
        sklearn_prediction.astype(np.int64),
        onnx_prediction.astype(np.int64)
    )

    print(
        "Predictions match:",
        prediction_match
    )

    # --------------------------------------------------------
    # Numerical tolerances
    # --------------------------------------------------------

    decision_match = decision_diff <= 1e-10
    probability_match = probability_diff <= 1e-10

    print(
        "Decision scores match:",
        decision_match
    )

    print(
        "Probabilities match:",
        probability_match
    )

    # --------------------------------------------------------
    # Detailed mismatches
    # --------------------------------------------------------

    if not prediction_match:

        print()
        print("PREDICTION MISMATCHES")

        for i in range(len(tests)):

            if (
                sklearn_prediction[i]
                != onnx_prediction[i]
            ):

                print(
                    i,
                    repr(tests[i]),
                    "SKLEARN=",
                    sklearn_prediction[i],
                    "ONNX=",
                    onnx_prediction[i]
                )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    passed = (
        decision_match
        and probability_match
        and prediction_match
    )

    print()
    print(
        "RESULT:",
        "PASS" if passed else "FAIL"
    )

    return passed


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SAFE-PLACE TOXIC DETECTOR V2")
    print("=" * 70)

    model = load_model()

    inspect_model(model)

    create_onnx_classifier(model)

    result = validate(model)

    print()
    print("=" * 70)
    print("FINAL STATUS")
    print("=" * 70)

    if result:
        print("toxic_detector_classifier: PASS")
        print("STATUS: PASS")
    else:
        print("toxic_detector_classifier: FAIL")
        print("STATUS: FAIL")


if __name__ == "__main__":
    main()