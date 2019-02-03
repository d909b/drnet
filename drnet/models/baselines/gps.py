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
from scipy.stats import norm
from functools import partial
from sklearn.decomposition import PCA
from drnet.models.baselines.baseline import Baseline


class GPS(Baseline):
    def __init__(self):
        super(GPS, self).__init__()
        self.gps = None
        self.pca = None

    def install_grf(self):
        from rpy2.robjects.packages import importr
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects.vectors import StrVector
        import rpy2.robjects as robjects

        # robjects.r.options(download_file_method='curl')
        # package_names = ["causaldrf"]
        # utils = rpackages.importr('utils')
        # utils.chooseCRANmirror(ind=0)
        # utils.chooseCRANmirror(ind=0)
        #
        # names_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
        # if len(names_to_install) > 0:
        #     utils.install_packages(StrVector(names_to_install))

        return importr("causaldrf")

    def _build(self, **kwargs):
        from rpy2.robjects import numpy2ri, pandas2ri
        gps = self.install_grf()

        self.gps = gps
        numpy2ri.activate()
        pandas2ri.activate()

        self.with_exposure = kwargs["with_exposure"]
        return [None for _ in range(kwargs["num_treatments"])]

    def predict(self, x):
        def get_x_by_idx(idx):
            data = [x[0][idx], x[1][idx]]
            if len(x) == 1:
                data[0] = np.expand_dims(data[0], axis=0)

            if self.with_exposure:
                data += [x[2][idx]]
            return data

        if self.pca is not None:
            x[0] = self.pca.transform(x[0])

        results = np.zeros((len(x[0],)))
        for treatment_idx in range(len(self.model)):
            indices = np.where(x[1] == treatment_idx)[0]
            this_x = get_x_by_idx(indices)
            y_pred = np.array(self.predict_for_model(self.model[treatment_idx], this_x))
            results[indices] = y_pred
        return results

    def predict_for_model(self, old_model, x):
        import rpy2.robjects as robjects
        from rpy2.robjects import Formula, pandas2ri

        distribution, model = old_model

        r = robjects.r
        gps = distribution.pdf(x[-1])
        data_frame = pandas2ri.py2ri(
            Baseline.to_data_frame(np.column_stack([x[-1], gps]), column_names=["T", "gps"])
        )
        result = r.predict(model, data_frame)
        return np.array(result)

    def fit_generator_for_model(self, model, train_generator, train_steps, val_generator, val_steps, num_epochs):
        all_outputs = []
        for _ in range(train_steps):
            generator_output = next(train_generator)
            x, y = generator_output[0], generator_output[1]
            all_outputs.append((x, y))
        x, y = zip(*all_outputs)
        x = map(partial(np.concatenate, axis=0), zip(*x))
        y = np.concatenate(y, axis=0)

        if x[0].shape[-1] > 200:
            self.pca = PCA(16, svd_solver="randomized")
            x[0] = self.pca.fit_transform(x[0])

        treatment_xy = self.split_by_treatment(x, y)
        for key in treatment_xy.keys():
            x, y = treatment_xy[key]
            self.model[int(key)] = self.build_drf_model(x, y)

    def build_drf_model(self, x_old, y):
        from rpy2.robjects.vectors import StrVector, FactorVector, FloatVector, IntVector
        from rpy2.robjects import Formula, pandas2ri

        x, ts = x_old[:, :-1], x_old[:, -1]

        tmp = np.concatenate([x, np.reshape(ts, (-1, 1)), np.reshape(y, (-1, 1))], axis=-1)
        data_frame = pandas2ri.py2ri(
            Baseline.to_data_frame(tmp, column_names=np.arange(0, tmp.shape[-1] - 2).tolist() + ["T", "Y"])
        )

        result = self.gps.hi_est(Y="Y",
                                 treat="T",
                                 treat_formula=Formula('T ~ ' + '+'.join(data_frame.names[:-2])),
                                 outcome_formula=Formula('Y ~ T + I(T^2) + gps + T * gps'),
                                 data=data_frame,
                                 grid_val=FloatVector([float(tt) for tt in np.linspace(0, 1, 256)]),
                                 treat_mod="Normal",
                                 link_function="log")  # link_function is not used with treat_mod = "Normal".

        treatment_model, model = result[1], result[2]
        fitted_values = treatment_model.rx2('fitted.values')
        distribution = norm(np.mean(fitted_values), np.std(fitted_values))
        return distribution, model

    def preprocess(self, x):
        if self.with_exposure:
            return np.concatenate([x[0], np.reshape(x[2], (-1, 1))], axis=-1)
        else:
            return x[0]

    def postprocess(self, y):
        return y[:, -1]
