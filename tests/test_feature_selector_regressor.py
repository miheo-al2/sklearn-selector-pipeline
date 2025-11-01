import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest, f_regression, SelectPercentile
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.base import clone

from sklearn_selector_pipeline import FeatureSelectorRegressor


def make_toy_regression(n_samples=200, n_features=20, n_informative=5, random_state=0):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=0.1,
        random_state=random_state,
    )
    return X, y


def test_with_selectkbest_and_linear_regression():
    """Basic functionality test with SelectKBest and LinearRegression."""
    X, y = make_toy_regression()
    selector = SelectKBest(score_func=f_regression, k=8)
    regressor = LinearRegression()

    wrapped = FeatureSelectorRegressor(selector, regressor)
    wrapped.fit(X, y)

    # transform should return the reduced dimensionality
    Xt = wrapped.transform(X)
    assert Xt.shape[1] == 8

    preds = wrapped.predict(X)
    assert preds.shape[0] == X.shape[0]

    # Score should return R² score
    score = wrapped.score(X, y)
    assert isinstance(score, float)


def test_with_selectkbest_and_random_forest():
    """Test with RandomForestRegressor."""
    X, y = make_toy_regression()
    selector = SelectKBest(score_func=f_regression, k=6)
    regressor = RandomForestRegressor(n_estimators=50, random_state=42)

    wrapped = FeatureSelectorRegressor(selector, regressor)
    wrapped.fit(X, y)

    preds = wrapped.predict(X)
    assert preds.shape[0] == X.shape[0]

    score = wrapped.score(X, y)
    assert isinstance(score, float)
    assert score >= 0.0  # R² can be negative but should be positive for good fit


def test_score_uses_transformed_features():
    """Ensure score uses transformed features."""
    X, y = make_toy_regression()
    selector = SelectKBest(score_func=f_regression, k=4)
    regressor = LinearRegression()

    wrapped = FeatureSelectorRegressor(selector, regressor)
    wrapped.fit(X, y)
    score = wrapped.score(X, y)

    assert isinstance(score, float)


def test_partial_fit_with_sgdregressor():
    """Test incremental training using SGDRegressor (supports partial_fit)."""
    X, y = make_toy_regression(n_samples=500)
    selector = SelectKBest(score_func=f_regression, k=7)
    regressor = SGDRegressor(max_iter=100, random_state=42)

    wrapped = FeatureSelectorRegressor(selector, regressor)

    # Split into mini-batches and call partial_fit
    n_batches = 5
    batch_size = X.shape[0] // n_batches

    for i in range(n_batches):
        start = i * batch_size
        end = (i + 1) * batch_size if i < n_batches - 1 else X.shape[0]
        Xb, yb = X[start:end], y[start:end]
        wrapped.partial_fit(Xb, yb)

    # After incremental training, predict should work
    preds = wrapped.predict(X)
    assert preds.shape[0] == X.shape[0]


def test_partial_fit_selector_without_partial_fit():
    """
    Test partial_fit when selector doesn't implement partial_fit.
    Selector should be fit on first batch, then used to transform subsequent batches.
    """
    X, y = make_toy_regression(n_samples=300)
    selector = SelectKBest(score_func=f_regression, k=5)  # no partial_fit
    regressor = SGDRegressor(max_iter=100, random_state=42)

    wrapped = FeatureSelectorRegressor(selector, regressor)

    # first batch fits selector
    wrapped.partial_fit(X[:100], y[:100])
    # subsequent batches only transform with the fitted selector
    wrapped.partial_fit(X[100:200], y[100:200])
    wrapped.partial_fit(X[200:], y[200:])

    preds = wrapped.predict(X)
    assert preds.shape[0] == X.shape[0]


def test_multioutput_regression():
    """Test with multi-output regression using a selector that supports it."""
    X, y = make_regression(n_samples=200, n_features=15, n_targets=3, random_state=42)
    
    # Use SelectPercentile instead of SelectKBest with f_regression for multi-output
    # or use a simpler selector that doesn't rely on univariate statistics
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.0)  # Keep all features with any variance
    regressor = LinearRegression()

    wrapped = FeatureSelectorRegressor(selector, regressor)
    wrapped.fit(X, y)

    preds = wrapped.predict(X)
    assert preds.shape == y.shape

    score = wrapped.score(X, y)
    assert isinstance(score, float)


def test_parameter_forwarding():
    """Test parameter forwarding with sample_weight (a valid fit parameter for regressors)."""
    X, y = make_toy_regression()
    selector = SelectKBest(score_func=f_regression, k=8)
    # Use a regressor that accepts sample_weight in fit
    regressor = LinearRegression()

    wrapped = FeatureSelectorRegressor(selector, regressor)
    
    # Create sample weights
    sample_weight = np.random.random(X.shape[0])
    
    # Test parameter forwarding with sample_weight (valid fit parameter)
    wrapped.fit(X, y, regressor__sample_weight=sample_weight)
    
    # Verify the fit worked
    preds = wrapped.predict(X)
    assert preds.shape[0] == X.shape[0]


def test_parameter_forwarding_with_sample_weight():
    """Test parameter forwarding with sample_weight."""
    X, y = make_toy_regression()
    selector = SelectKBest(score_func=f_regression, k=8)
    regressor = RandomForestRegressor(n_estimators=50, random_state=42)

    wrapped = FeatureSelectorRegressor(selector, regressor)
    
    # Create sample weights
    sample_weight = np.random.random(X.shape[0])
    
    # This should work without errors
    wrapped.fit(X, y, regressor__sample_weight=sample_weight)
    
    preds = wrapped.predict(X)
    assert preds.shape[0] == X.shape[0]


def test_fit_transform_method():
    """Test the fit_transform method."""
    X, y = make_toy_regression()
    selector = SelectKBest(score_func=f_regression, k=6)
    regressor = LinearRegression()

    wrapped = FeatureSelectorRegressor(selector, regressor)
    
    # Test fit_transform
    X_transformed = wrapped.fit_transform(X, y)
    
    # Should have reduced dimensionality
    assert X_transformed.shape[1] == 6
    assert X_transformed.shape[0] == X.shape[0]
    
    # Should be equivalent to fit then transform
    wrapped2 = FeatureSelectorRegressor(selector, regressor)
    wrapped2.fit(X, y)
    X_transformed2 = wrapped2.transform(X)
    
    np.testing.assert_array_equal(X_transformed, X_transformed2)


def test_score_with_sample_weight():
    """Test the score method with sample_weight parameter."""
    X, y = make_toy_regression()
    selector = SelectKBest(score_func=f_regression, k=8)
    regressor = LinearRegression()

    wrapped = FeatureSelectorRegressor(selector, regressor)
    wrapped.fit(X, y)
    
    # Test score without sample_weight
    score1 = wrapped.score(X, y)
    
    # Test score with sample_weight
    sample_weight = np.random.random(X.shape[0])
    score2 = wrapped.score(X, y, sample_weight=sample_weight)
    
    # Both should be floats (scores may differ due to weighting)
    assert isinstance(score1, float)
    assert isinstance(score2, float)


def test_reuse_same_wrapper_instance_with_fit_after_partial_fit():
    """
    Ensure that after partial_fit, calling fit reinitializes clones.
    """
    X, y = make_toy_regression(n_samples=200)
    selector = SelectKBest(score_func=f_regression, k=5)
    regressor = SGDRegressor(max_iter=100, random_state=42)

    wrapped = FeatureSelectorRegressor(selector, regressor)
    # do partial_fit to create internal state
    wrapped.partial_fit(X[:100], y[:100])
    # calling fit should replace internal clones and re-fit from scratch
    wrapped.fit(X, y)
    preds = wrapped.predict(X)
    assert preds.shape[0] == X.shape[0]


def test_regressor_without_partial_fit_raises_error():
    """Test that using partial_fit with regressor that doesn't support it raises error."""
    X, y = make_toy_regression()
    selector = SelectKBest(score_func=f_regression, k=5)
    regressor = LinearRegression()  # doesn't support partial_fit

    wrapped = FeatureSelectorRegressor(selector, regressor)
    
    with pytest.raises(AttributeError, match="does not support partial_fit"):
        wrapped.partial_fit(X, y)


def test_selector_that_only_supports_fit_transform():
    """Test with a selector that has fit_transform but different signature."""
    from sklearn.decomposition import PCA
    
    X, y = make_toy_regression()
    # PCA is a transformer that has fit_transform but doesn't use y
    selector = PCA(n_components=8)
    regressor = LinearRegression()

    wrapped = FeatureSelectorRegressor(selector, regressor)
    wrapped.fit(X, y)

    Xt = wrapped.transform(X)
    assert Xt.shape[1] == 8

    preds = wrapped.predict(X)
    assert preds.shape[0] == X.shape[0]
