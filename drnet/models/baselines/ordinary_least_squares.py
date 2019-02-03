"""
Copyright (C) 2019  Patrick Schwab, ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import numpy as np
from functools import partial
from sklearn.linear_model import LinearRegression, LogisticRegression
from drnet.models.baselines.baseline import Baseline, PickleableMixin


class OrdinaryLeastSquares1(PickleableMixin, Baseline):
    def __init__(self):
        super(OrdinaryLeastSquares1, self).__init__()

    def _build(self, **kwargs):
        self.with_exposure = kwargs["with_exposure"]
        return LinearRegression()

    def preprocess(self, x):
        if self.with_exposure:
            return np.concatenate([x[0], np.expand_dims(x[1], axis=-1), np.expand_dims(x[2], axis=-1)], axis=-1)
        else:
            return np.concatenate([x[0], np.expand_dims(x[1], axis=-1)], axis=-1)

    def postprocess(self, y):
        if y.ndim > 1:
            return y[:, -1]
        else:
            return y


class OrdinaryLeastSquares2(PickleableMixin, Baseline):
    def __init__(self):
        super(OrdinaryLeastSquares2, self).__init__()

    def _build(self, **kwargs):
        num_treatments = kwargs["num_treatments"]
        self.with_exposure = kwargs["with_exposure"]
        return [LinearRegression() for _ in range(num_treatments)]

    def preprocess(self, x):
        if self.with_exposure:
            return np.concatenate([x[0], np.reshape(x[2], (-1, 1))], axis=-1)
        else:
            return x[0]

    def postprocess(self, y):
        if y.ndim > 1:
            return y[:, -1]
        else:
            return y

    def predict(self, x):
        def get_x_by_idx(idx):
            data = [np.expand_dims(x[0][idx], axis=0), x[1][idx]]
            if self.with_exposure:
                data += [x[2][idx]]
            return data

        return np.array(map(lambda idx: self.predict_for_model(self.model[x[1][idx]],
                                                               get_x_by_idx(idx)),
                            np.arange(len(x[1]))))

    def fit_generator(self, train_generator, train_steps, val_generator, val_steps, num_epochs, batch_size):
        all_outputs = []
        for _ in range(train_steps):
            generator_output = next(train_generator)
            x, y = generator_output[0], generator_output[1]
            all_outputs.append((x, y))
        x, y = zip(*all_outputs)
        x = map(partial(np.concatenate, axis=0), zip(*x))
        y = np.concatenate(y, axis=0)

        treatment_xy = self.split_by_treatment(x, y)
        for key in treatment_xy.keys():
            x, y = treatment_xy[key]
            self.model[int(key)].fit(x, y)
