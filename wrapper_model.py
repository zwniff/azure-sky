from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
import catboost as ctb

class XGBWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = xgb.XGBClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

# Wrapper for CatBoost
class CTBWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = ctb.CatBoostClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)