from .data import Data
from .sampler import BatchSampler
import numpy as np
import torch


class Quadruple(Data):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.train_x = X_train
        self.train_y = y_train
        self.test_x = X_test
        self.test_y = y_test

        self.train_sampler = BatchSampler(len(self.train_y), shuffle=True)
        self.time_sampler = BatchSampler(len(self.train_x[2]), shuffle=True)

        self.num_time = len(self.train_x[2])
        self.nx = 96
        self.ny = 200
        self.space_dim = self.nx * self.ny

    def losses(self, targets, outputs, indices, istrain, loss_fn, inputs, model, aux=None):
        if not torch.is_tensor(targets):
            targets = torch.as_tensor(targets, device=outputs.device, dtype=outputs.dtype)
        else:
            targets = targets.to(device=outputs.device, dtype=outputs.dtype)
        return loss_fn(targets, outputs)

    def _slice_time_outputs(self, y, time_indices):
        y = y.reshape(y.shape[0], self.num_time, self.space_dim)
        y = y[:, time_indices, :]
        y = y.reshape(y.shape[0], -1)
        return y

    def train_next_batch(self, batch_size=None, timestep_batch_size=None, training_time_size=None):
        if batch_size is None:
            train_indices = np.arange(len(self.train_y))
            return self.train_x, self.train_y, train_indices

        sample_indices = self.train_sampler.get_next(batch_size)

        x1 = self.train_x[0][sample_indices]
        x2 = self.train_x[1][sample_indices]
        xt = self.train_x[2]
        y = self.train_y[sample_indices]

        if timestep_batch_size is not None:
            time_indices = self.time_sampler.get_next(timestep_batch_size)
            time_indices = np.sort(time_indices)
            xt = xt[time_indices]
            y = self._slice_time_outputs(y, time_indices)
        else:
            time_indices = np.arange(self.num_time)

        train_indices = {
            "sample_indices": sample_indices,
            "time_indices": time_indices,
        }

        return (x1, x2, xt), y, train_indices

    def test(self):
        test_indices = {
            "sample_indices": np.arange(len(self.test_y)),
            "time_indices": np.arange(len(self.test_x[2])),
        }
        return self.test_x, self.test_y, test_indices


class QuadrupleCartesianProd(Data):
    def __init__(self, X_train, y_train, X_test, y_test):
        if (
            len(X_train[0]) * len(X_train[2]) != y_train.size
            or len(X_train[1]) * len(X_train[2]) != y_train.size
            or len(X_train[0]) != len(X_train[1])
        ):
            raise ValueError("The training dataset does not have the format of Cartesian product.")

        if (
            len(X_test[0]) * len(X_test[2]) != y_test.size
            or len(X_test[1]) * len(X_test[2]) != y_test.size
            or len(X_test[0]) != len(X_test[1])
        ):
            raise ValueError("The testing dataset does not have the format of Cartesian product.")

        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test
        self.branch_sampler = BatchSampler(len(X_train[0]), shuffle=True)
        self.trunk_sampler = BatchSampler(len(X_train[2]), shuffle=True)

    def losses(self, targets, outputs, indices, istrain, loss_fn, inputs, model, aux=None):
        if not torch.is_tensor(targets):
            targets = torch.as_tensor(targets, device=outputs.device, dtype=outputs.dtype)
        else:
            targets = targets.to(device=outputs.device, dtype=outputs.dtype)
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None, timestep_batch_size=None, training_time_size=None):
        if batch_size is None:
            train_indices = {
                "branch_indices": np.arange(len(self.train_x[0])),
                "trunk_indices": np.arange(len(self.train_x[2])),
            }
            return self.train_x, self.train_y, train_indices

        if not isinstance(batch_size, (tuple, list)):
            indices = self.branch_sampler.get_next(batch_size)
            train_indices = {
                "branch_indices": indices,
                "trunk_indices": np.arange(len(self.train_x[2])),
            }
            return (
                self.train_x[0][indices],
                self.train_x[1][indices],
                self.train_x[2],
            ), self.train_y[indices], train_indices

        indices_branch = self.branch_sampler.get_next(batch_size[0])
        indices_trunk = self.trunk_sampler.get_next(batch_size[1])
        train_indices = {
            "branch_indices": indices_branch,
            "trunk_indices": indices_trunk,
        }
        return (
            self.train_x[0][indices_branch],
            self.train_x[1][indices_branch],
            self.train_x[2][indices_trunk],
        ), self.train_y[indices_branch][:, indices_trunk], train_indices

    def test(self):
        test_indices = {
            "branch_indices": np.arange(len(self.test_x[0])),
            "trunk_indices": np.arange(len(self.test_x[2])),
        }
        return self.test_x, self.test_y, test_indices