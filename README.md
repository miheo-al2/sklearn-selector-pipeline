# sklearn-selector-pipeline

A scikit-learn compatible package that provides meta-estimators for seamlessly combining feature selectors with classifiers and regressors into single pipeline components.

## Features

- 🔧 **Seamless Integration**: Works with any sklearn-compatible feature selector and classifier/regressor
- 🚀 **Full sklearn API**: Supports `fit`, `predict`, `predict_proba`, `decision_function`, `score`, and `transform`
- 📊 **Incremental Learning**: Supports `partial_fit` for online learning scenarios
- 🎯 **Parameter Forwarding**: Forward fit parameters to selector and classifier/regressor using prefixes
- 🔄 **Pipeline Compatible**: Can be used inside sklearn pipelines
- 🧪 **Extensively Tested**: Comprehensive test suite ensuring reliability
- 📈 **Dual Support**: Separate classes for classification and regression tasks

## Installation

```bash
pip install sklearn-selector-pipeline
```

For development installation:
```bash
pip install sklearn-selector-pipeline[dev]
```

## Quick Start

### Classification Example

```python
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn_selector_pipeline import FeatureSelectorClassifier

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the meta-estimator
selector = SelectKBest(score_func=f_classif, k=10)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
meta_clf = FeatureSelectorClassifier(feature_selector=selector, classifier=classifier)

# Fit and predict
meta_clf.fit(X_train, y_train)
predictions = meta_clf.predict(X_test)
probabilities = meta_clf.predict_proba(X_test)
accuracy = meta_clf.score(X_test, y_test)

print(f"Accuracy: {accuracy:.3f}")
print(f"Selected features shape: {meta_clf.transform(X_test).shape}")
```

### Regression Example

```python
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn_selector_pipeline import FeatureSelectorRegressor

# Generate sample data
X, y = make_regression(n_samples=1000, n_features=20, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the meta-estimator
selector = SelectKBest(score_func=f_regression, k=10)
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
meta_reg = FeatureSelectorRegressor(feature_selector=selector, regressor=regressor)

# Fit and predict
meta_reg.fit(X_train, y_train)
predictions = meta_reg.predict(X_test)
r2_score = meta_reg.score(X_test, y_test)

print(f"R² Score: {r2_score:.3f}")
print(f"Selected features shape: {meta_reg.transform(X_test).shape}")
```

## Advanced Usage

### Parameter Forwarding

Use prefixes to pass parameters specifically to the selector or classifier/regressor:

```python
# Classification
meta_clf.fit(X_train, y_train, 
             selector__k=15,  # parameter for SelectKBest
             classifier__sample_weight=sample_weights)  # parameter for classifier

# Regression  
meta_reg.fit(X_train, y_train,
             selector__k=8,  # parameter for SelectKBest
             regressor__sample_weight=sample_weights)  # parameter for regressor
```

### Partial Fit for Online Learning

```python
from sklearn.linear_model import SGDClassifier, SGDRegressor

# Classification with online learning
selector = SelectKBest(k=10)
online_clf = SGDClassifier()
meta_clf = FeatureSelectorClassifier(selector, online_clf)

for X_batch, y_batch in data_batches:
    meta_clf.partial_fit(X_batch, y_batch, classes=np.unique(y))

# Regression with online learning  
selector = SelectKBest(k=10)
online_reg = SGDRegressor()
meta_reg = FeatureSelectorRegressor(selector, online_reg)

for X_batch, y_batch in data_batches:
    meta_reg.partial_fit(X_batch, y_batch)
```

### Usage in Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Classification pipeline
clf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_clf', FeatureSelectorClassifier(selector, classifier))
])

# Regression pipeline
reg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_reg', FeatureSelectorRegressor(selector, regressor))
])
```

## API Reference

### FeatureSelectorClassifier

**Parameters:**
- `feature_selector`: Any sklearn-compatible feature selector
- `classifier`: Any sklearn-compatible classifier

**Methods:**
- `fit(X, y, **fit_params)`: Fit the selector then the classifier
- `predict(X)`: Make predictions
- `predict_proba(X)`: Predict class probabilities (if supported)
- `decision_function(X)`: Get decision function values (if supported)
- `transform(X)`: Transform features using the fitted selector
- `score(X, y)`: Return accuracy score
- `partial_fit(X, y, classes=None, **fit_params)`: Incremental fit

### FeatureSelectorRegressor

**Parameters:**
- `feature_selector`: Any sklearn-compatible feature selector
- `regressor`: Any sklearn-compatible regressor

**Methods:**
- `fit(X, y, **fit_params)`: Fit the selector then the regressor
- `predict(X)`: Make predictions
- `transform(X)`: Transform features using the fitted selector
- `score(X, y)`: Return R² score
- `partial_fit(X, y, **fit_params)`: Incremental fit

## Examples

Check out the `examples/` directory for comprehensive examples:
- Basic classification and regression usage
- Genetic Algorithm feature selector example
- Evaluation on real datasets with visualization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Citation

```bibtex
@software{sklearn_selector_pipeline,
  author = {Debajyati},
  title = {sklearn-selector-pipeline: Meta-estimators for combining feature selectors with classifiers and regressors},
  url = {https://github.com/Debajyati/sklearn-selector-pipeline},
  version = {0.1.2},
  year = {2025}
}
```
