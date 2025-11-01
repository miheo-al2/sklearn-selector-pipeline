import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.base import clone

from sklearn_selector_pipeline import FeatureSelectorClassifier


def make_toy(n_samples=200, n_features=20, n_informative=5, random_state=0):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        random_state=random_state,
    )
    return X, y


def test_with_selectkbest_and_logistic_predict_and_proba():
    """Classifier that supports predict_proba should work via wrapper."""
    X, y = make_toy()
    selector = SelectKBest(score_func=f_classif, k=8)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")

    wrapped = FeatureSelectorClassifier(selector, clf)
    wrapped.fit(X, y)

    # transform should return the reduced dimensionality
    Xt = wrapped.transform(X)
    assert Xt.shape[1] == 8

    preds = wrapped.predict(X)
    assert preds.shape[0] == X.shape[0]

    probs = wrapped.predict_proba(X)
    assert probs.shape[0] == X.shape[0]
    # probability columns must match number of classes
    assert probs.shape[1] == len(np.unique(y))


def test_with_selectkbest_and_svm_no_proba():
    """Classifier without predict_proba should raise when calling predict_proba."""
    X, y = make_toy()
    selector = SelectKBest(score_func=f_classif, k=6)
    # LinearSVC doesn't implement predict_proba by default
    clf = LinearSVC(max_iter=2000)

    wrapped = FeatureSelectorClassifier(selector, clf)
    wrapped.fit(X, y)

    with pytest.raises(AttributeError):
        wrapped.predict_proba(X)

    # But predict should work
    preds = wrapped.predict(X)
    assert preds.shape[0] == X.shape[0]


def test_score_uses_transformed_features():
    X, y = make_toy()
    selector = SelectKBest(score_func=f_classif, k=4)
    clf = LogisticRegression(max_iter=1000)

    wrapped = FeatureSelectorClassifier(selector, clf)
    wrapped.fit(X, y)
    s = wrapped.score(X, y)

    # Score should be a float in [0,1]
    assert isinstance(s, float)
    assert 0.0 <= s <= 1.0


def test_partial_fit_with_sgdclassifier():
    """Test incremental training using SGDClassifier (supports partial_fit)."""
    X, y = make_toy(n_samples=500)
    selector = SelectKBest(score_func=f_classif, k=7)
    clf = SGDClassifier(max_iter=100)

    wrapped = FeatureSelectorClassifier(selector, clf)

    # Split into mini-batches and call partial_fit
    n_batches = 5
    batch_size = X.shape[0] // n_batches

    # Provide classes on first call to partial_fit to be safe
    classes = np.unique(y)
    for i in range(n_batches):
        start = i * batch_size
        end = (i + 1) * batch_size if i < n_batches - 1 else X.shape[0]
        Xb, yb = X[start:end], y[start:end]
        wrapped.partial_fit(Xb, yb, classes=classes)

    # After incremental training, predict should work
    preds = wrapped.predict(X)
    assert preds.shape[0] == X.shape[0]


def test_partial_fit_selector_without_partial_fit():
    """
    Ensure that if the selector doesn't implement partial_fit, the wrapper still works:
    selector is fit on the first batch then used to transform subsequent batches.
    """
    X, y = make_toy(n_samples=300)
    selector = SelectKBest(score_func=f_classif, k=5)  # no partial_fit
    clf = SGDClassifier(max_iter=500)

    wrapped = FeatureSelectorClassifier(selector, clf)

    # first batch fits selector
    wrapped.partial_fit(X[:100], y[:100], classes=np.unique(y))
    # subsequent batches only transform with the fitted selector
    wrapped.partial_fit(X[100:200], y[100:200])
    wrapped.partial_fit(X[200:], y[200:])

    preds = wrapped.predict(X)
    assert preds.shape[0] == X.shape[0]


def test_reuse_same_wrapper_instance_with_fit_after_partial_fit():
    """
    Ensure that after partial_fit, calling fit reinitializes clones (fresh fit behavior).
    """
    X, y = make_toy(n_samples=200)
    selector = SelectKBest(score_func=f_classif, k=5)
    clf = SGDClassifier(max_iter=500)

    wrapped = FeatureSelectorClassifier(selector, clf)
    # do partial_fit to create internal state
    wrapped.partial_fit(X[:100], y[:100], classes=np.unique(y))
    # calling fit should replace internal clones and re-fit from scratch
    wrapped.fit(X, y)
    preds = wrapped.predict(X)
    assert preds.shape[0] == X.shape[0]

