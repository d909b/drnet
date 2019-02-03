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

import sqlite3
import numpy as np
from os.path import join
from drnet.data_access.data_access import DataAccess as BaseDataAccess


class DataAccess(BaseDataAccess):
    DB_FILE_NAME = "icu_windows.db"
    TABLE_PATIENTS = "patients"
    TABLE_WINDOWS = "windows"
    TABLE_OUTCOMES = "outcomes"

    def __init__(self, data_dir, is_factual, flatten_inputs=True):
        self.db = self.connect(data_dir)
        self.is_factual = is_factual
        self.setup_schema()
        self.cache_no_rowid = {}
        self.all_signals = ["cns_ti-na", "cns_te-na",
                            "cns_ftotal-na", "cns_etco2-na",
                            "cns_spo2-na", "cns_icp-mean", "cns_tinfinity-a",
                            "cns_tinfinity-b", "cns_rr-na", "cns_hr-na",
                            "cns_cstat-na", "cns_expminvol-na", "cns_fspontpct-na",
                            "cns_pinsp-na", "cns_pmean-na", "cns_pminimum-na",
                            "cns_ppeak-na", "cns_pplateau-na", "cns_rinsp-na",
                            "cns_art-mean",
                            "cns_art-syst",
                            "cns_art-dias",
                            "cns_ap-dias",
                            "cns_ap-syst",
                            "cns_ap-mean",
                            "cns_cvp-mean",
                            "cns_cpp-na",
                            "cns_ci-na", "cns_co-na", "cns_cpi-na", "cns_cpo-na",
                            "cns_ict-na", "cns_pbto2-na", "cns_dpmx-na", "cns_elwi-na", "cns_evlw-na",
                            "cns_gedi-na", "cns_gedv-na", "cns_gef-na", "cns_itbi-na", "cns_itbv-na", "cns_pcco-na",
                            "cns_ppv-na",
                            "cns_pvpi-na", "cns_sv-na", "cns_svv-na", "cns_svi-na", "cns_svr-na"
        ]

        self.input_dimensions = None
        self.flatten_inputs = flatten_inputs

    def connect(self, data_dir):
        db = sqlite3.connect(join(data_dir, DataAccess.DB_FILE_NAME),
                             check_same_thread=False,
                             detect_types=sqlite3.PARSE_DECLTYPES)
        # Indices:
        # create index w01 on windows (signal_name);
        # create index w02 on windows (window_id);
        # create index w03 on windows (window_id,signal_name);
        # create index p01 on patients (patient_id);
        return db

    def setup_schema(self):
        self.db.execute("CREATE TABLE IF NOT EXISTS {table_name} "
                        "("
                        "patient_id TEXT NOT NULL, "
                        "window_id INT, "
                        "PRIMARY KEY (patient_id, window_id)"
                        ");".format(table_name=DataAccess.TABLE_PATIENTS))

        self.db.execute("CREATE TABLE IF NOT EXISTS {table_name} "
                        "("
                        "window_id INT NOT NULL,"
                        "signal_name TEXT NOT NULL,"
                        "timestamps ARRAY,"
                        "vals ARRAY,"
                        "sampling_rate INT NOT NULL,"
                        "window_length_in_seconds INT NOT NULL,"
                        "source_db TEXT NOT NULL,"
                        "PRIMARY KEY (window_id, signal_name)"
                        ");"
                        .format(table_name=DataAccess.TABLE_WINDOWS))

        self.db.execute("CREATE TABLE IF NOT EXISTS {table_name} "
                        "("
                        "window_id INT NOT NULL,"
                        "outcome_name TEXT NOT NULL,"
                        "timestamp INT NOT NULL,"
                        "value FLOAT,"
                        "unit TEXT NOT NULL,"
                        "PRIMARY KEY (window_id, outcome_name)"
                        ");"
                        .format(table_name=DataAccess.TABLE_OUTCOMES))

        self.db.commit()

    def has_table(self, table_name):
        return self.db.execute("SELECT name "
                               "FROM sqlite_master "
                               "WHERE type='table' AND name='{table_name}';"
                               .format(table_name=table_name))\
                   .fetchone() is not None

    def get_all_signal_types(self):
        return map(lambda x: x[0], self.db.execute("SELECT DISTINCT signal_name FROM {table_name};"
                                                   .format(table_name=DataAccess.TABLE_WINDOWS)).fetchall())

    def get_input_dimensions(self, allow_flattened=True):
        if self.input_dimensions is None:
            if False and self.is_factual:
                input_dimensions = []
                for signal_name in self.all_signals:
                    sampling_rate, window_length_in_seconds = self.db.execute("SELECT sampling_rate, window_length_in_seconds "
                                                                              "FROM {windows_table} "
                                                                              "WHERE signal_name = ? LIMIT 1;"
                                                                              .format(windows_table=DataAccess.TABLE_WINDOWS),
                                                                              (signal_name,)).fetchone()
                    input_dimensions.append((int(sampling_rate*window_length_in_seconds),))
            else:
                input_dimensions = [(1,) for _ in range(len(self.all_signals))]
            self.input_dimensions = input_dimensions

        returned_input_dimensions = self.input_dimensions
        if self.flatten_inputs and allow_flattened:
            returned_input_dimensions = (int(sum(map(lambda x: x[0], self.input_dimensions))),)
        return returned_input_dimensions

    def get_num_rows(self, table_name):
        # This query assumes that there has not been any deletions in the time series table.
        return self.db.execute("SELECT MAX(_ROWID_) "
                               "FROM '{table_name}' "
                               "LIMIT 1;"
                               .format(table_name=table_name))\
                      .fetchone()[0]

    def insert_many_values(self, table_name, values):
        self.db.executemany("INSERT OR IGNORE INTO '{table_name}' "
                            "VALUES ({question_marks});"
                            .format(table_name=table_name,
                                    question_marks=",".join(['?']*len(values[0]))),
                            values)

    def get_all_table_names(self):
        return map(lambda x: x[0], self.db.execute("SELECT name FROM 'sqlite_master' WHERE type='table';").fetchall())

    def get_patients(self):
        patients_list_of_tuples = self.db.execute("SELECT DISTINCT patient_id "
                                                  "FROM {patients_table} "
                                                  "ORDER BY _ROWID_;"
                                                  .format(patients_table=DataAccess.TABLE_PATIENTS)).fetchall()
        return map(lambda x: x[0], patients_list_of_tuples)

    def get_last_timestamp(self, table_name):
        result = self.db.execute("SELECT timestamp "
                                 "FROM '{table_name}' "
                                 "WHERE _ROWID_ = (SELECT MAX(_ROWID_) FROM '{table_name}');"
                                 .format(table_name=table_name)).fetchone()

        return result if result is None else result[0]

    def get_labelled_patients(self):
        return np.arange(self.get_num_rows(DataAccess.TABLE_OUTCOMES)) + 1

    def get_labels(self, args, ids, benchmark):
        assignments = []
        for i, id in enumerate(ids):
            data = self.get_row(DataAccess.TABLE_WINDOWS, id[0], with_rowid=False)
            assignment = benchmark.get_assignment(id[0], data[0])[0]
            assignments.append(assignment)
        assignments = np.array(assignments)
        num_labels = benchmark.get_num_treatments()
        return assignments, num_labels

    def get_entry_with_id(self, id, args={}):
        data = self.get_row(DataAccess.TABLE_WINDOWS, id, with_rowid=True)

        patient_id = data[0]
        result = {"id": patient_id, "x": data[1]}
        return patient_id, result

    def get_row(self, table_name, id, columns=None, with_rowid=False,
                do_cache=True, zero_impute=True, normalize=True):
        if id in self.cache_no_rowid and do_cache:
            return_value = self.cache_no_rowid[id]
            if with_rowid:
                return id, return_value
            else:
                return return_value

        is_signals = False
        if columns is None:
            is_signals = True
            columns = "signal_name, vals"

        if with_rowid:
            columns = "rowid, " + columns

        query = "SELECT " \
                "{columns} " \
                "FROM {table_name} " \
                "WHERE window_id = ?;".format(table_name=table_name,
                                              columns=columns)

        return_value = self.db.execute(query, (id,)).fetchall()

        if is_signals:
            signal_data = {}
            for result_tuple in return_value:
                if with_rowid:
                    _, signal_name, values = result_tuple
                else:
                    signal_name, values = result_tuple

                if values is not None:
                    signal_data[signal_name] = values

            # Use __all_signals__ order.
            return_value = []
            for signal_name, expected_dimensions in zip(self.all_signals,
                                                        self.get_input_dimensions(allow_flattened=False)):
                if signal_name in signal_data:
                    if normalize:
                        value_range = DataAccess.get_natural_value_range(signal_name)
                        if value_range is None:
                            print("missing value range:", signal_name)
                            value_range = [0, 100]
                        min_value, max_value = value_range
                        data = (signal_data[signal_name] - min_value) / float(max_value - min_value)
                        data = np.clip(data, 0.0, 1.0)
                    else:
                        data = signal_data[signal_name]

                    if data.shape != expected_dimensions:
                        # Ensure expected dimension matches with given dimensions.
                        data = data[-expected_dimensions[0]:]
                    return_value.append(data)
                else:
                    if zero_impute:
                        return_value.append(np.zeros(expected_dimensions))
                    else:
                        return_value.append(None)

            if self.flatten_inputs:
                return_value = np.hstack(return_value)

            if do_cache:
                self.cache_no_rowid[id] = return_value

        if with_rowid:
            return id, return_value
        else:
            return return_value

    def get_rows(self, train_ids, columns="rowid, x"):
        results = []
        for train_id in train_ids:
            results.append(self.get_row(DataAccess.TABLE_WINDOWS, train_id, with_rowid=True))

        ids = np.array(map(lambda x: x[0], results))
        signal_data = map(lambda x: x[1], results)
        signal_data = np.array(signal_data)
        return signal_data, ids, signal_data

    @staticmethod
    def get_natural_value_range(device_and_signal_name):
        limits_dict = {
            "cns_art-mean": [0, 200],
            "cns_art-syst": [0, 300],
            "cns_art-dias": [0, 200],
            "draeger_art": [0, 2000],
            "draeger_cvp": [0, 150],
            "cns_ap-na": [0, 200],
            "cns_ap-dias": [0, 200],
            "cns_ap-syst": [0, 200],
            "cns_ap-mean": [0, 200],
            "cns_cvp-mean": [0, 15],
            "cns_cpp-na": [0, 180],
            "cns_cpp2-na": [0, 180],
            "draeger_icp": [0, 300],
            "draeger_icp2": [0, 300],
            "cns_icp-mean": [0, 100],
            "cns_icp2-mean": [0, 100],
            "cns_ci-na": [0, 10],
            "cns_co-na": [0, 20],
            "cns_cpi-na": [0, 10],
            "cns_cpo-na": [0, 20],
            "cns_hr-na": [0, 300],
            "cns_lap-mean": [0, 100],
            "cns_rr-na": [0, 60],
            "cns_spo2-na": [0, 100],
            "draeger_spo2": [0, 100],
            "draeger_resp": [0, 50],
            "cns_cstat-na": [0, 500],
            "cns_etco2-na": [0, 200],
            "cns_expminvol-na": [0, 40],
            "cns_fio2-na": [0, 100],
            "cns_ftotal-na": [0, 50],
            "cns_peep-na": [5, 30],
            "cns_pinsp-na": [0, 100],
            "cns_pmean-na": [0, 100],
            "cns_pminimum-na": [0, 300],
            "cns_ppeak-na": [0, 100],
            "cns_pplateau-na": [0, 100],
            "cns_rinsp-na": [0, 50],
            "cns_rsb-na": [0, 100],
            "cns_te-na": [0, 10],
            "cns_ti-na": [0, 10],
            "cns_fspontpct-na": [0, 100],
            "cns_vte-na": [0, 1000],
            "cns_tinfinity-a": [0, 40],
            "cns_tinfinity-b": [0, 40],
            "cns_ict-na": [0, 100],
            "cns_pbto2-na": [0, 100],
            "cns_dpmx-na": [0, 1000],
            "cns_elwi-na": [0, 30],
            "cns_evlw-na": [0, 2000],
            "cns_gedi-na": [0, 2000],
            "cns_gedv-na": [0, 5000],
            "cns_gef-na": [0, 100],
            "cns_itbi-na": [0, 3000],
            "cns_itbv-na": [0, 6000],
            "cns_pcco-na": [0, 30],
            "cns_ppv-na": [0, 100],
            "cns_pvpi-na": [0, 10],
            "cns_sv-na": [0, 500],
            "cns_svv-na": [0, 100],
            "cns_svi-na": [0, 200],
            "cns_svr-na": [0, 6000],
            "cns_nbp-mean": [0, 100]
        }
        if device_and_signal_name in limits_dict:
            return limits_dict[device_and_signal_name]
        else:
            return None