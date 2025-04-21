from src.samplers._samplers import InfluenceSampler, AbstractSampler
from src.training_models import get_dataloader
from src.FastIF import *
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

        # dataloader to compute estimations of hvp
        batch_train_data_loader = get_dataloader(x, y, batch_size=kwargs["batch_size"], random=True)
        # two dataloaders to compute influences
        instance_train_data_loader = get_dataloader(x, y, batch_size=1, random=False)
        eval_instance_data_loader = get_dataloader(x_eval, y_eval, batch_size=1, random=False)
        # number of test (validation) points for which Influence Function will be calculated
        num_examples_to_test = kwargs["num_examples_to_test"] if "num_examples_to_test" in kwargs else x.shape[0] // 4
        # number of batches on which hvp is calculated
        s_test_num_samples = min(x.shape[0] // kwargs["batch_size"] - 1, 1000)

        influences = np.zeros(shape=[num_examples_to_test, x.shape[0]])
        num_examples_tested = 0

        if "use_knn" in kwargs and kwargs["use_knn"]:
            index = FAISSIndex(x.shape[-1])
            index.add(x)
            nearest_neighbors = index.search(
                k=kwargs["knn_k_value"] if "knn_k_value" in kwargs else x.shape[0] // 10,
                queries=x_eval[:num_examples_to_test]
            )[1]
        else:
            nearest_neighbors = ["all"] * num_examples_to_test
        with tqdm(total=num_examples_to_test) as pbar:
            for test_index, test_inputs in enumerate(eval_instance_data_loader):
                if num_examples_tested >= num_examples_to_test:
                    break

                influences[test_index] = compute_influences(
                    batch_train_data_loader=batch_train_data_loader,
                    instance_train_data_loader=instance_train_data_loader,
                    model=self.model,
                    test_inputs=test_inputs,
                    s_test_num_samples=s_test_num_samples,
                    s_test_iterations=1,
                    train_indices_to_include=nearest_neighbors[test_index],
                    fill_value=-np.inf,
                )

                num_examples_tested += 1
                pbar.update(1)

        # IFs are calculated just for k nearest neighbors for each test instance so we have a lot of empty cells in
        # influence array. They are filled with -infinity so max function would ignore them. Sampler chooses
        # train points with the least influence values (i.e. the least harmful) so -infinity changed to +infinity
        # (just in case IF wasn't calculated for some train points).
        inf_test = np.nan_to_num(influences.max(axis=0), nan=np.inf, neginf=np.inf)
        # TODO: add possibility to change function 'max' by user's parameters

        return self.sampler(x, y, weight, influences=inf_test)
