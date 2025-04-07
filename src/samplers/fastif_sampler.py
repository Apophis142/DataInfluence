from src.samplers._samplers import InfluenceSampler, AbstractSampler
from src.FastIF.fast_influence_utils import run_full_influence_functions
from src.learning_models import get_dataloader
import numpy as np


class FastIFSampler(AbstractSampler):
    def __init__(self, num_samples: int, model):
        self.model = model
        self.sampler = InfluenceSampler(num_samples)
        self.num_samples = num_samples

    def __call__(self, x, y, weight, *args, **kwargs):
        if (any(it not in kwargs for it in ["x_eval", "y_eval", "batch_size"])
                or not isinstance(kwargs["batch_size"], int)):
            raise ValueError

        x_eval = kwargs["x_eval"]
        y_eval = kwargs["y_eval"]

        influences = run_full_influence_functions(
            self.model,
            get_dataloader(x, y, batch_size=kwargs["batch_size"], random=True),
            get_dataloader(x, y, batch_size=1, random=False),
            get_dataloader(x_eval, y_eval, batch_size=1, random=False),
            num_examples_to_test=kwargs["num_examples_to_test"] if "num_examples_to_test" in kwargs else x.shape[0] // 4,
            s_test_num_samples=min(x.shape[0] // kwargs["batch_size"] - 1, 1000)
        )
        # TODO: add kNN implementation

        inf_test = np.array(
            [list(val["influences"].values()) for val in influences.values()]
        ).max(axis=0)
        # TODO: add possibility to change function 'max' by user's parameters

        return self.sampler(x, y, weight, influences=inf_test)
