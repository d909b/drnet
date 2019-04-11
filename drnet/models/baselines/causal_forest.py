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
from drnet.models.baselines.baseline import Baseline, PickleableMixin


class CausalForest(PickleableMixin, Baseline):
    def __init__(self):
        super(CausalForest, self).__init__()
        self.bart = None

    def install_grf(self):
        from rpy2.robjects.packages import importr
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects.vectors import StrVector
        import rpy2.robjects as robjects

        # robjects.r.options(download_file_method='curl')

        # package_names = ["grf"]
        # utils = rpackages.importr('utils')
        # utils.chooseCRANmirror(ind=0)
        # utils.chooseCRANmirror(ind=0)
        #
        # names_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
        # if len(names_to_install) > 0:
        #     utils.install_packages(StrVector(names_to_install))

        return importr("grf")

    def _build(self, **kwargs):
        from rpy2.robjects import numpy2ri
        from sklearn import linear_model
        grf = self.install_grf()

        self.grf = grf
        numpy2ri.activate()
        num_treatments = kwargs["num_treatments"]
        self.with_exposure = kwargs["with_exposure"]

        return [linear_model.Ridge(alpha=.5)] +\
               [None for _ in range(num_treatments)]

    def predict_for_model(self, model, x):
        base_y = Baseline.predict_for_model(self, self.model[0], x)
        if model == self.model[0]:
            return base_y
        else:
            import rpy2.robjects as robjects
            r = robjects.r
            result = r.predict(model, self.preprocess(x))
            y = np.array(result[0])
            return y[:, -1] + base_y

    def fit_grf_model(self, x, t, y):
        from rpy2.robjects.vectors import StrVector, FactorVector, FloatVector, IntVector
        return self.grf.causal_forest(x,
                                      FloatVector([float(yy) for yy in y]),
                                      FloatVector([float(tt) for tt in t]), seed=909)

    def preprocess(self, x):
        if self.with_exposure:
            return np.concatenate([x[0], np.reshape(x[2], (-1, 1))], axis=-1)
        else:
            return x[0]

    def postprocess(self, y):
        return y

    def predict(self, x):
        def get_x_by_idx(idx):
            data = map(lambda x_inner: x_inner[idx], x)
            if len(x) == 1:
                data[0] = np.expand_dims(data[0], axis=0)

            return data

        results = np.zeros((len(x[0], )))
        for treatment_idx in range(len(self.model)):
            indices = np.where(x[1] == treatment_idx)[0]

            if len(indices) == 0:
                continue

            this_x = get_x_by_idx(indices)
            y_pred = np.array(self.predict_for_model(self.model[treatment_idx], this_x))
            results[indices] = y_pred
        return results

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
        x_c, y_c = treatment_xy[0]
        self.model[0].fit(x_c, y_c)

        for key in treatment_xy.keys():
            if key == 0:
                continue

            x_i, y_i = treatment_xy[key]
            x, y = np.concatenate([x_c, x_i], axis=0), np.concatenate([y_c, y_i], axis=0)
            t = [0]*len(x_c) + [1]*len(x_i)
            self.model[int(key)] = self.fit_grf_model(x, t, y)

    def split_by_treatment(self, x, y):
        treatment_xy = {}
        for i in range(len(self.model)):
            indices = filter(lambda idx: x[1][idx] == i, np.arange(len(x[0])))
            treatment_xy[i] = (self.preprocess(map(lambda x_inner: x_inner[indices], x)), y[indices])
        return treatment_xy
