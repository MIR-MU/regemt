from typing import Iterable, List, Tuple, Optional, Any
from functools import lru_cache
import warnings

from common import ReferenceFreeMetric, Judgements
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import RegressorMixin as Model
from sklearn.feature_selection import RFECV
from sklearn.linear_model import (
    LinearRegression,
    SGDRegressor,
    Ridge,
    BayesianRidge,
)
from sklearn.utils import parallel_backend
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from tqdm.autonotebook import tqdm


Feature = float
Features = Tuple[Feature, ...]

Scores = List[float]


class Regression(ReferenceFreeMetric):

    label = "Regression"
    model: Optional[Model] = None
    judgements: Optional[Judgements] = None

    def __init__(self, metrics: Iterable[ReferenceFreeMetric], reference_free: bool = False):
        if reference_free:
            metrics = [metric for metric in metrics if isinstance(metric, ReferenceFreeMetric)]

        self.metrics = tuple(metrics)
        self.reference_free = reference_free

    def _get_metric_features(self, judgements: Judgements) -> List[Features]:
        metric_features_transposed = []
        for metric in self.metrics:
            if self.reference_free:
                results = metric.compute_ref_free(judgements)
            else:
                results = metric.compute(judgements)
            are_finite = np.isfinite(results)
            if not np.all(are_finite):
                num_non_finite = len(results) - np.sum(are_finite)
                message = f'{num_non_finite} out of {len(results)} results returned by {metric} are not finite'
                raise ValueError(message)
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
        return list(judgements.scores)

    def _get_models(self, select_features: bool = True,
                    optimize_hyperparameters: bool = True,
                    random_state: float = 42) -> Model:

        def linear_regression():
            return {
                'model': LinearRegression(),
                'hyperparameters': {
                    'normalize': [True, False],
                    'positive': [True, False],
                },
                'can_select_features': True,
            }

        def sgd_regressor():
            return {
                'model': SGDRegressor(random_state=random_state),
                'hyperparameters': {
                    'penalty': ['l2', 'l1', 'elasticnet'],
                    'early_stopping': [True, False],
                },
                'can_select_features': True,
            }

        def ridge():
            return {
                'model': Ridge(random_state=random_state),
                'hyperparameters': {
                    'alpha': np.logspace(1, 4, 10),
                },
                'can_select_features': True,
            }

        def bayesian_ridge():
            return {
                'model': BayesianRidge(),
                'hyperparameters': None,
                'can_select_features': True,
            }

        def svr():
            return {
                'model': LinearSVR(dual=False, loss='squared_epsilon_insensitive'),
                'hyperparameters': {
                    'C': np.logspace(-2, 3, 10),
                },
                'can_select_features': True,
            }

        def k_nearest_neighbors_regressor():
            return {
                'model': KNeighborsRegressor(),
                'hyperparameters': {
                    'n_neighbors': range(1, 20, 2),
                },
                'can_select_features': False,
            }

        def mlp_regressor():
            return {
                'model': MLPRegressor(random_state=random_state),
                'hyperparameters': {
                    'solver': ['lbfgs', 'sgd', 'adam'],
                    'alpha': np.logspace(1, 4, 10),
                },
                'can_select_features': False,
            }

        models = [
            linear_regression(),
            sgd_regressor(),
            ridge(),
            bayesian_ridge(),
            svr(),
            k_nearest_neighbors_regressor(),
            mlp_regressor(),
        ]

        for model in tqdm(models, desc=f'{self}: model selection'):
            estimator = model['model']
            if optimize_hyperparameters and model['hyperparameters'] is not None:
                estimator = GridSearchCV(estimator, model['hyperparameters'])
            if select_features and model['can_select_features']:
                estimator = RFECV(estimator)
            yield make_pipeline(StandardScaler(), estimator)

    def fit(self, judgements: Judgements):
        print(f'{self}: getting features on train judgements')
        X, y = self._get_features(judgements), self._get_scores(judgements)

        train_judgements, test_judgements = judgements.split()
        train_X, train_y = X[:len(train_judgements)], y[:len(train_judgements)]
        test_X, test_y = X[len(train_judgements):], y[len(train_judgements):]
        assert (len(train_X), len(train_y)) == (len(train_judgements), len(train_judgements))
        assert (len(test_X), len(test_y)) == (len(test_judgements), len(test_judgements))

        models, best_model, best_r2 = self._get_models(), None, float('-inf')
        with parallel_backend('multiprocessing', n_jobs=-1), warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            for model in models:
                model.fit(train_X, train_y)
                r2 = model.score(test_X, test_y)
                if r2 > best_r2:
                    best_model, best_r2 = model, r2
        assert best_model is not None
        print(f'{self}: selected model {best_model}')

        print(f'{self}: fitting the selected model')
        best_model.fit(X, y)

        self.judgements = judgements
        self.model = best_model

    @lru_cache(maxsize=None)
    def compute(self, judgements: Judgements) -> List[float]:
        if self.model is None:
            raise ValueError('Using compute() before fit()')
        if self.judgements.overlaps(judgements):
            raise ValueError('Train and test judgements overlap')

        print(f'{self}: getting features on test judgements')
        X = self._get_features(judgements)

        print(f'{self}: making predictions with the selected model')
        y = self.model.predict(X)
        return y

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Regression):
            return NotImplemented
        return all([
            self.reference_free == other.reference_free,
            self.metrics == other.metrics,
            self.judgements == other.judgements,
        ])

    def __hash__(self) -> int:
        return hash((self.reference_free, self.metrics, self.judgements))
