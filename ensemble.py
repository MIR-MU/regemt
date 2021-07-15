from typing import Iterable, List, Tuple, Optional

from common import ReferenceFreeMetric, Judgements
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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from tqdm.autonotebook import tqdm


Feature = float
Features = Tuple[Feature, ...]


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

    def _get_models(self) -> Iterable[Model]:
        models = [
            LinearRegression(),
            SGDRegressor(),
            Ridge(),
            BayesianRidge(),
            SVR(kernel='rbf'),
            KNeighborsRegressor(),
            PLSRegression(),
            MLPRegressor(),
        ]
        for model in tqdm(models, desc=f'{self.label}: model selection'):
            yield make_pipeline(StandardScaler(), model)

    def fit(self, judgements: Judgements):
        train_judgements, test_judgements = judgements.split()

        train_X, train_y = self._get_features(train_judgements), train_judgements.scores
        test_X, true_test_y = self._get_features(test_judgements), test_judgements.scores
        models, best_model, best_mse = self._get_models(), None, float('inf')
        for model in models:
            model.fit(train_X, train_y)
            predicted_test_y = model.predict(test_X)
            mse = mean_squared_error(true_test_y, predicted_test_y)
            if mse < best_mse:
                best_model, best_mse = model, mse
        assert best_model is not None

        X, y = self._get_features(judgements), judgements.scores
        best_model.fit(X, y)

        self.model = best_model

    def compute(self, judgements: Judgements) -> List[float]:
        if self.model is None:
            raise ValueError('Using compute() before fit()')

        X = self._get_features(judgements)
        y = self.model.predict(X)
        return y
