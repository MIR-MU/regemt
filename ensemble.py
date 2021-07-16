from typing import Iterable, List, Tuple, Optional

from common import ReferenceFreeMetric, Judgements
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import RegressorMixin as Model
from sklearn.linear_model import (
    LinearRegression,
    SGDRegressor,
    Ridge,
    BayesianRidge,
)
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from tqdm.autonotebook import tqdm


Feature = float
Features = Tuple[Feature, ...]

Scores = List[float]


class Regression(ReferenceFreeMetric):

    label = "Regression"
    model: Optional[Model] = None

    def __init__(self, metrics: Iterable[ReferenceFreeMetric], reference_free: bool = False):
        if reference_free:
            metrics = [metric for metric in metrics if isinstance(metric, ReferenceFreeMetric)]

        self.metrics = list(metrics)
        self.reference_free = reference_free

    def _get_metric_features(self, judgements: Judgements) -> List[Features]:
        metric_features_transposed = []
        for metric in self.metrics:
            if self.reference_free:
                results = metric.compute_ref_free(judgements)
            else:
                results = metric.compute(judgements)
            metric_features_transposed.append(results)
        metric_features = list(zip(*metric_features_transposed))
        return metric_features

    def _get_other_features(self, judgements: Judgements) -> List[Features]:
        other_features = []
        sources = judgements.src_texts if self.reference_free else [t[0] for t in judgements.references]
        for source, translation in zip(sources, judgements.translations):
            other_features.append((float(len(source)), float(len(translation))))
        return other_features

    def _get_features(self, judgements: Judgements) -> List[Features]:
        metric_features_list = self._get_metric_features(judgements)
        other_features_list = self._get_other_features(judgements)
        features = []
        for metric_features, other_features in zip(metric_features_list, other_features_list):
            features.append((*metric_features, *other_features))
        return features

    def _get_scores(self, judgements: Judgements) -> Scores:
        return judgements.scores

    def _get_models(self, random_state: float = 42) -> Model:
        def linear_regression() -> Model:
            return make_pipeline(
                StandardScaler(),
                GridSearchCV(
                    LinearRegression(),
                    {
                        'normalize': [True, False],
                        'positive': [True, False],
                    },
                    n_jobs=-1,
                )
            )

        def sgd_regressor() -> Model:
            return make_pipeline(
                StandardScaler(),
                GridSearchCV(
                    SGDRegressor(random_state=random_state),
                    {
                        'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                        'penalty': ['l2', 'l1', 'elasticnet'],
                        'early_stopping': [True, False],
                    },
                    n_jobs=-1,
                )
            )

        def ridge() -> Model:
            return make_pipeline(
                StandardScaler(),
                GridSearchCV(
                    Ridge(random_state=random_state),
                    {
                        'alpha': np.logspace(1, 4, 50),
                    },
                    n_jobs=-1,
                )
            )

        def bayesian_ridge() -> Model:
            return make_pipeline(
                StandardScaler(),
                BayesianRidge(),
            )

        def svr() -> Model:
            return make_pipeline(
                StandardScaler(),
                GridSearchCV(
                    SVR(),
                    {
                        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                        'C': np.logspace(-2, 3, 50),
                    },
                    n_jobs=-1,
                )
            )

        def k_nearest_neighbors_regressor() -> Model:
            return make_pipeline(
                StandardScaler(),
                GridSearchCV(
                    KNeighborsRegressor(),
                    {
                        'n_neighbors': range(1, 20, 2),
                    },
                    n_jobs=-1,
                )
            )

        def mlp_regressor() -> Model:
            return make_pipeline(
                StandardScaler(),
                GridSearchCV(
                    MLPRegressor(random_state=random_state),
                    {
                        'activation': ['identity', 'logistic', 'tanh', 'relu'],
                        'solver': ['lbfgs', 'sgd', 'adam'],
                        'alpha': np.logspace(1, 4, 50),
                    },
                    n_jobs=-1,
                )
            )

        models = [
            linear_regression(),
            sgd_regressor(),
            ridge(),
            bayesian_ridge(),
            svr(),
            k_nearest_neighbors_regressor(),
            mlp_regressor(),
        ]

        for model in tqdm(models, desc=f'{self.label}: model selection'):
            yield model

    def fit(self, judgements: Judgements):
        train_judgements, test_judgements = judgements.split()

        train_X, train_y = self._get_features(train_judgements), self._get_scores(train_judgements)
        test_X, test_y = self._get_features(test_judgements), self._get_scores(test_judgements)
        models, best_model, best_r2 = self._get_models(), None, float('-inf')
        for model in models:
            model.fit(train_X, train_y)
            r2 = model.score(test_X, test_y)
            if r2 > best_r2:
                best_model, best_r2 = model, r2
        assert best_model is not None

        X, y = self._get_features(judgements), self._get_scores(judgements)
        best_model.fit(X, y)

        self.model = best_model

    def compute(self, judgements: Judgements) -> List[float]:
        if self.model is None:
            raise ValueError('Using compute() before fit()')

        X = self._get_features(judgements)
        y = self.model.predict(X)
        return y
