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
from __future__ import print_function

import sys
import numpy as np
from sklearn.svm import SVC
from bisect import bisect_left
from sklearn.decomposition import PCA


class PropensityBatch(object):
    def propensity_list_is_initialised(self):
        return self.treatment_lists is not None

    def make_propensity_lists(self, train_ids, benchmark_implementation,
                              reduce_dimensionality=True,
                              dimensionality_reduction_limit=200,
                              dimensionality_reduction_target=50,
                              with_exposure=False,
                              **kwargs):
        from drnet.models.benchmarks.tcga_benchmark import TCGABenchmark

        input_data, ids, pair_data = benchmark_implementation.get_data_access().get_rows(train_ids)
        assignments = map(benchmark_implementation.get_assignment, ids, input_data)
        values = zip(*assignments)
        if with_exposure:
            treatment_data, batch_y, treatment_strength = values
        else:
            treatment_data, batch_y = values

        treatment_data = np.array(treatment_data)

        if isinstance(benchmark_implementation, TCGABenchmark):
            pair_data = benchmark_implementation.select_features(pair_data)

        if pair_data.shape[-1] > dimensionality_reduction_limit and reduce_dimensionality:
            self.pca = PCA(dimensionality_reduction_target, svd_solver="randomized")
            pair_data = self.pca.fit_transform(pair_data)
        else:
            self.pca = None

        self.clf = SVC(probability=True, class_weight="balanced")
        self.clf.fit(pair_data, treatment_data)
        propensity_scores = self.clf.predict_proba(pair_data)

        linked_data = zip(input_data, propensity_scores, ids)

        self.treatment_lists = []
        for treatment_idx in range(benchmark_implementation.get_num_treatments()):
            this_treatment_data = [linked_data[idx] for idx in np.where(treatment_data == treatment_idx)[0]]
            self.treatment_lists.append(sorted(this_treatment_data, key=lambda x: x[1][treatment_idx]))
        print("INFO: Prepared propensity lists.", file=sys.stderr)

    def get_closest_in_propensity_lists(self, x, t, k):
        if self.pca is None:
            propensity_score = self.clf.predict_proba(x.reshape(1, -1))[0, t]
        else:
            propensity_score = self.clf.predict_proba(self.pca.transform(x.reshape(1, -1)))[0, t]

        class TreatmentListWrapper:
            def __init__(self, obj):
                self.obj = obj

            def __getitem__(self, key):
                return self.obj[key][1][t]

            def __len__(self):
                return len(self.obj)

        idx = bisect_left(TreatmentListWrapper(self.treatment_lists[t]), propensity_score)
        if idx == len(self.treatment_lists[t]):
            idx -= 1

        if idx != 0:
            idx = max(0, idx - np.random.randint(0, k))

        return self.treatment_lists[t][idx][0], self.treatment_lists[t][idx][-1]

    def enhance_batch_with_propensity_matches(self, args, benchmark, treatment_data, input_data, batch_y,
                                              treatment_strengths, match_probability=1.0, num_randomised_neighbours=6):
        all_matches = []
        for treatment_idx in range(benchmark.get_num_treatments()):
            this_treatment_indices = np.where(treatment_data == treatment_idx)[0]
            matches = map(lambda t:
                          map(lambda idx: self.get_closest_in_propensity_lists(input_data[idx], t,
                                                                               k=num_randomised_neighbours),
                              this_treatment_indices),
                          [t_idx for t_idx in range(benchmark.get_num_treatments()) if t_idx != treatment_idx])
            all_matches += reduce(lambda x, y: x + y, matches)

        if match_probability != 1.0:
            all_matches = np.random.permutation(all_matches)[:int(len(all_matches)*match_probability)]

        match_ids = map(lambda x: x[1], all_matches)
        all_matches = np.array(map(lambda x: x[0], all_matches))

        from drnet.models.benchmarks.twins_benchmark import TwinsBenchmark
        from drnet.models.benchmarks.tcga_benchmark import TCGABenchmark
        if isinstance(benchmark, TwinsBenchmark):
            match_input_data = all_matches[:, 7:]
        elif isinstance(benchmark, TCGABenchmark):
            match_input_data = benchmark.select_features(all_matches)
        else:
            match_input_data = all_matches

        match_assignments = map(benchmark.get_assignment, match_ids, all_matches)

        if args["with_exposure"]:
            match_treatment_data, match_batch_y, match_strengths = zip(*match_assignments)
            treatment_strengths = np.concatenate([treatment_strengths, match_strengths], axis=0)
        else:
            match_treatment_data, match_batch_y = zip(*match_assignments)
            treatment_strengths = None

        return np.concatenate([input_data, match_input_data], axis=0), \
               np.concatenate([treatment_data, match_treatment_data], axis=0), \
               np.concatenate([batch_y, match_batch_y], axis=0), \
               treatment_strengths
