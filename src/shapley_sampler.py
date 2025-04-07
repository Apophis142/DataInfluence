import numpy as np
from pydvl.value import compute_shapley_values
from pydvl.utils import Dataset, Utility
from pydvl.value.shapley import ShapleyMode, MaxUpdates
from sklearn.ensemble import HistGradientBoostingRegressor
from src.samplers import AbstractSampler, IndexSampler


class ShapleySampler(AbstractSampler):
    def __init__(self, X_test, y_test, num_samples=100):
        self.num_samples = num_samples
        self.X_test = X_test
        self.y_test = y_test
        self.index_sampler = IndexSampler()

    def __call__(self, X, y, weight):
        dataset = Dataset(x_train=X, y_train=y, x_test=self.X_test, y_test=self.y_test)
        model = HistGradientBoostingRegressor(max_iter=50, max_depth=3)
        utility = Utility(model, dataset)

        shapley_values_tmc = compute_shapley_values(
            utility,
            mode=ShapleyMode.TruncatedMontecarlo,
            done=MaxUpdates(10),
            n_jobs=1,
            progress=True
        )

        indices = np.argsort(shapley_values_tmc.values)[-self.num_samples:]

        return self.index_sampler(X, y, weight, index=indices)