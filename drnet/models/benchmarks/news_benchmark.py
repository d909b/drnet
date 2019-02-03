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
from drnet.apps.parameters import clip_percentage
from drnet.apps.util import stable_softmax, gaussian
from sklearn.metrics.pairwise import euclidean_distances
from drnet.models.benchmarks.benchmark import Benchmark
from drnet.data_access.news.data_access import DataAccess


class NewsBenchmark(Benchmark):
    def __init__(self, data_dir,
                 response_mean_of_mean=0.45, response_std_of_mean=0.15,
                 response_mean_of_std=0.1, response_std_of_std=0.05,
                 strength_of_assignment_bias=10, epsilon_std=0.15,
                 num_samples=5000, num_treatments=16,
                 with_exposure=True, **kwargs):
        super(NewsBenchmark, self).__init__(DataAccess(data_dir), num_treatments, **kwargs)
        self.centroids = None
        self.dosage_centroids = None
        self.assignment_cache = {}
        self.response_mean_of_mean = response_mean_of_mean
        self.response_std_of_mean = response_std_of_mean
        self.response_mean_of_std = response_mean_of_std
        self.response_std_of_std = response_std_of_std
        self.strength_of_assignment_bias = strength_of_assignment_bias
        self.epsilon_std = epsilon_std
        self.seed = kwargs["seed"]
        self.random_generator = None
        self.num_samples = num_samples
        self.with_exposure = with_exposure
        self.num_archetypes_per_treatment = 2
        self.scaling_constant = 50

    def get_scaling_constant(self):
        return self.scaling_constant

    def has_exposure(self):
        return self.with_exposure

    def filter(self, patients):
        return self.random_generator.choice(patients, size=self.num_samples, replace=False)

    def get_input_shapes(self, args):
        return (self.data_access.get_news_dimension(),)

    def get_output_shapes(self, args):
        return (1,)

    def initialise(self, args, seed=909):
        self.random_generator = np.random.RandomState(seed)
        self.centroids = None

    def get_from_generator_with_offsets(self, generator, centroid_indices, adjust_last=False):
        from drnet.data_access.generator import get_last_id_set

        centroids_tmp, current_idx = [], 0
        while len(centroid_indices) != 0:
            x, _ = next(generator)
            ids = get_last_id_set()

            while len(centroid_indices) != 0 and centroid_indices[0] <= current_idx + len(x[0]):
                next_index = centroid_indices[0]
                del centroid_indices[0]

                is_last_treatment = len(centroid_indices) == 0
                if is_last_treatment and adjust_last:
                    # Last treatment is control = worse expected outcomes.
                    response_mean_of_mean = 1 - self.response_mean_of_mean
                else:
                    response_mean_of_mean = self.response_mean_of_mean

                response_mean = clip_percentage(self.random_generator.normal(response_mean_of_mean,
                                                                             self.response_std_of_mean))
                response_std = clip_percentage(self.random_generator.normal(self.response_mean_of_std,
                                                                            self.response_std_of_std)) + 0.025

                id, data = self.data_access.get_entry_with_id(ids[next_index])
                z = data["z"]

                centroids_tmp.append((z, response_mean, response_std))
            current_idx += len(x[0])
        return centroids_tmp

    def fit(self, generator, steps, batch_size):
        num_samples = steps*batch_size
        centroid_indices = sorted(self.random_generator.permutation(num_samples)[:self.num_treatments + 1])

        if self.with_exposure:
            self.dosage_centroids = []
            for treatment_idx in range(self.num_treatments):
                dosage_centroid_indices = \
                    sorted(self.random_generator.permutation(num_samples)[:self.num_archetypes_per_treatment])
                self.dosage_centroids.append(self.get_from_generator_with_offsets(generator, dosage_centroid_indices))
                for dosage_idx in range(self.num_archetypes_per_treatment):
                    min_response = self.random_generator.normal(0.0, 0.1)
                    self.dosage_centroids[treatment_idx][dosage_idx] += (min_response,)
        self.centroids = self.get_from_generator_with_offsets(generator, centroid_indices, adjust_last=True)
        self.assignment_cache = {}

    def get_assignment(self, id, x):
        if self.centroids is None:
            if self.with_exposure:
                return 0, 0, 0
            else:
                return 0, 0

        if id not in self.assignment_cache:
            id, data = self.data_access.get_entry_with_id(id)
            z = data["z"]
            values = self._assign(z)
            self.assignment_cache[id] = values

        if self.with_exposure:
            assigned_treatment, assigned_y, treatment_strength = self.assignment_cache[id]

            if self.assign_counterfactuals:
                return assigned_treatment, assigned_y, treatment_strength
            else:
                return assigned_treatment, assigned_y[assigned_treatment], treatment_strength[assigned_treatment]
        else:
            assigned_treatment, assigned_y = self.assignment_cache[id]

            if self.assign_counterfactuals:
                return assigned_treatment, assigned_y
            else:
                return assigned_treatment, assigned_y[assigned_treatment]

    def get_dose_response_curve(self, z, treatment_idx, return_all=False):
        dosage_distances = self.get_centroid_weights(z, centroids=self.dosage_centroids[treatment_idx])
        normalised_distances = stable_softmax(self.strength_of_assignment_bias * dosage_distances)
        d = normalised_distances
        _, d0_mean, d0_std, d0_min = self.dosage_centroids[treatment_idx][0]
        _, d1_mean, d1_std, d1_min = self.dosage_centroids[treatment_idx][1]

        def dose_response_curve(treatment_strength):
            this_y = d[0] * gaussian(treatment_strength - d0_min, d0_mean, d0_std) + \
                     d[1] * gaussian(treatment_strength - d1_min, d1_mean, d1_std)
            return this_y

        if return_all:
            return dose_response_curve, d, d0_mean, d0_std, d0_min, d1_mean, d1_std, d1_min
        else:
            return dose_response_curve

    def _assign(self, x):
        # Assignment should be biased towards treatments that help more.
        assert self.centroids is not None, "Must call __fit__ before __assign__."

        distances = self.get_centroid_weights(x)

        expected_responses = []
        for treatment in range(self.num_treatments + 1):
            _, response_mean, response_std = self.centroids[treatment]
            y_this_treatment = self.random_generator.normal(response_mean, response_std)
            expected_responses.append(
                clip_percentage(y_this_treatment + self.random_generator.normal(0.0, self.epsilon_std))
            )
        expected_responses = np.array(expected_responses)
        control_response, control_distance = expected_responses[-1], distances[-1]

        y = []
        if self.with_exposure:
            treatment_strengths = []
            for treatment_idx in range(self.num_treatments):
                dose_response_curve, d, d0_mean, d0_std, d0_min, d1_mean, d1_std, d1_min \
                    = self.get_dose_response_curve(x,
                                                   treatment_idx,
                                                   return_all=True)

                treatment_strength = clip_percentage(
                    self.random_generator.exponential(scale=0.25)
                )

                treatment_strengths.append(treatment_strength)
                this_y = dose_response_curve(treatment_strength)
                y.append(this_y * expected_responses[treatment_idx])
            treatment_strengths = np.array(treatment_strengths)
        else:
            for treatment_idx in range(self.num_treatments):
                this_response, this_distance = expected_responses[treatment_idx], distances[treatment_idx]
                y.append(this_response * (this_distance + control_distance))
        y = np.array(y)

        treatment_chosen = self.random_generator.choice(self.num_treatments,
                                                        p=stable_softmax(
                                                            self.strength_of_assignment_bias * y)
                                                        )

        if self.with_exposure:
            return treatment_chosen, self.scaling_constant*y, treatment_strengths
        else:
            return treatment_chosen, self.scaling_constant*y

    def get_centroid_weights(self, x, centroids=None):
        if centroids is None:
            centroids = self.centroids

        similarities = map(
            lambda centroid: euclidean_distances(self.data_access.standardise_entry(x).reshape((1, -1)),
                                                 centroid.reshape((1, -1))),
            map(lambda x: x[0], centroids)
        )
        return np.squeeze(similarities)
