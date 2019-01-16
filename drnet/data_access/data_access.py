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
import abc
import numpy as np
from drnet.data_access.batch_augmentation import BatchAugmentation


class DataAccess(BatchAugmentation):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    def get_split_indices(self):
        return None, None

    def standardise_entry(self, entry):
        return entry

    @abc.abstractmethod
    def get_row(self, table_name, id, columns="x", with_rowid=False):
        pass

    @abc.abstractmethod
    def get_rows(self, train_ids, columns="rowid, x"):
        pass

    @abc.abstractmethod
    def get_labelled_patients(self):
        pass

    @abc.abstractmethod
    def get_labels(self, args, ids, benchmark):
        pass

    @abc.abstractmethod
    def get_entry_with_id(self, id, args):
        pass

    def prepare_batch(self, args, batch_data, benchmark, is_train=False):
        with_exposure = args["with_exposure"]
        ids = np.array(map(lambda x: x["id"], batch_data))
        ihdp_data = map(lambda x: x["x"], batch_data)

        assignments = map(benchmark.get_assignment, ids, ihdp_data)
        if with_exposure:
            treatment_data, batch_y, treatment_strength = zip(*assignments)
        else:
            treatment_data, batch_y = zip(*assignments)
            treatment_strength = None

        treatment_data = np.array(treatment_data)

        if args["with_propensity_batch"] and is_train:
            propensity_batch_probability = float(args["propensity_batch_probability"])
            num_randomised_neighbours = int(np.rint(args["num_randomised_neighbours"]))
            ihdp_data, treatment_data, batch_y, treatment_strength =\
                self.enhance_batch_with_propensity_matches(args,
                                                           benchmark,
                                                           treatment_data,
                                                           ihdp_data,
                                                           batch_y,
                                                           treatment_strength,
                                                           propensity_batch_probability,
                                                           num_randomised_neighbours)

        if isinstance(ihdp_data, list) and isinstance(ihdp_data[0], list):
            input_data = map(np.stack, zip(*ihdp_data))
            batch_x = input_data + [treatment_data]
        else:
            input_data = np.asarray(ihdp_data).astype(np.float32)
            batch_x = [
                input_data,
                treatment_data,
            ]

        if with_exposure:
            batch_x += [np.array(treatment_strength)]

        batch_y = np.array(batch_y)
        return batch_x, batch_y
