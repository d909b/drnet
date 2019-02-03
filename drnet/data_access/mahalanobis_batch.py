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

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree


class MahalanobisBatch(object):
    def propensity_list_is_initialised(self):
        return self.ball_trees is not None

    def make_propensity_lists(self, train_ids, benchmark_implementation,
                              reduce_dimensionality=False,
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
        treatment_data, ids = np.array(treatment_data), np.array(ids)

        if isinstance(benchmark_implementation, TCGABenchmark):
            pair_data = benchmark_implementation.select_features(pair_data)

        if pair_data.shape[-1] > dimensionality_reduction_limit and reduce_dimensionality:
            self.pca = PCA(dimensionality_reduction_target, svd_solver="randomized")
            pair_data = self.pca.fit_transform(pair_data)
        else:
            self.pca = None

        if with_exposure:
            pair_data = np.concatenate((pair_data, np.expand_dims(treatment_strength, axis=-1)), axis=-1)

        self.original_data = [pair_data[treatment_data == t]
                              for t in range(benchmark_implementation.get_num_treatments())]
        self.ball_trees = [BallTree(pair_data[treatment_data == t])
                           for t in range(benchmark_implementation.get_num_treatments())]
        self.treatment_ids = [ids[treatment_data == t]
                              for t in range(benchmark_implementation.get_num_treatments())]

    def get_closest_in_propensity_lists(self, x, t, k, treatment_strength=None):
        max_k = self.ball_trees[t].data.shape[0]
        adjusted_k = min(k, max_k)

        if self.pca is not None:
            x = self.pca.transform(x.reshape(1, -1))
        else:
            x = x.reshape(1, -1)

        if treatment_strength is not None:
            x = np.concatenate((x, np.reshape(treatment_strength, (1, 1))), axis=-1)

        distance, indices = self.ball_trees[t].query(x, k=adjusted_k)

        idx = np.random.randint(0, len(indices))
        idx = indices[0][idx]

        chosen_sample, chosen_id = self.original_data[t][idx], self.treatment_ids[t][idx]

        if treatment_strength is not None:
            chosen_sample = chosen_sample[:-1]

        return chosen_sample, chosen_id

    def enhance_batch_with_propensity_matches(self, args, benchmark, treatment_data, input_data, batch_y,
                                              treatment_strengths, match_probability=1.0, num_randomised_neighbours=6):
        all_matches = []
        for treatment_idx in range(benchmark.get_num_treatments()):
            this_treatment_indices = np.where(treatment_data == treatment_idx)[0]
            matches = map(lambda t:
                          map(lambda idx: self.get_closest_in_propensity_lists(input_data[idx], t,
                                                                               k=num_randomised_neighbours,
                                                                               treatment_strength=
                                                                               None if treatment_strengths is None else
                                                                               treatment_strengths[idx]),
                              this_treatment_indices),
                          [t_idx for t_idx in range(benchmark.get_num_treatments())
                           if t_idx != treatment_idx])

            if len(matches) != 0:
                all_matches += reduce(lambda x, y: x + y, matches)

        if match_probability != 1.0:
            all_matches = np.random.permutation(all_matches)[:int(len(all_matches)*match_probability)]
        match_ids = map(lambda x: x[1], all_matches)
        all_matches = np.array(map(lambda x: x[0], all_matches))

        from drnet.models.benchmarks.twins_benchmark import TwinsBenchmark
        if isinstance(benchmark, TwinsBenchmark):
            match_input_data = all_matches[:, 7:]
        else:
            match_input_data = all_matches

        # match_input_data = match_input_data + np.random.normal(0, 0.1, size=match_input_data.shape)

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
