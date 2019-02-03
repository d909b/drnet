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
from sklearn.ensemble import RandomForestRegressor
from drnet.models.baselines.baseline import Baseline, PickleableMixin


class RandomForest(PickleableMixin, Baseline):
    def __init__(self):
        super(RandomForest, self).__init__()

    def _build(self, **kwargs):
        num_units = int(np.rint(kwargs["num_units"]))
        num_layers = int(np.rint(kwargs["num_layers"]))
        self.with_exposure = kwargs["with_exposure"]
        return RandomForestRegressor(n_estimators=num_units, max_depth=num_layers)

    def preprocess(self, x):
        if self.with_exposure:
            return np.concatenate([x[0], np.reshape(x[1], (-1, 1)), np.reshape(x[2], (-1, 1))], axis=-1)
        else:
            return np.concatenate([x[0], np.reshape(x[1], (-1, 1))], axis=-1)

    def postprocess(self, y):
        if y.ndim > 1:
            return y[:, -1]
        else:
            return y
