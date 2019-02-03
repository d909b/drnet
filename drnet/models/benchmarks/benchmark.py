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


class Benchmark(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_access, num_treatments, **kwargs):
        self.data_access = data_access
        self.num_treatments = num_treatments
        self.assign_counterfactuals = False

    def filter(self, patients):
        return patients

    def get_num_treatments(self):
        return self.num_treatments

    def get_data_access(self):
        return self.data_access

    def set_assign_counterfactuals(self, value):
        self.assign_counterfactuals = value

    def has_exposure(self):
        return False

    @abc.abstractmethod
    def get_input_shapes(self, args):
        return (1,)

    @abc.abstractmethod
    def get_output_shapes(self, args):
        return (1,)

    @abc.abstractmethod
    def initialise(self, args):
        pass

    @abc.abstractmethod
    def fit(self, generator, steps, batch_size):
        pass

