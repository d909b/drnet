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
from os.path import join


class DataAccess(object):
    DB_FILE_NAME = "mimic3.db"
    TABLE_PATIENTS = "patients"
    TABLE_WINDOWS = "windows"
    TABLE_OUTCOMES = "outcomes"

    def __init__(self, data_dir):
        self.db = self.connect(data_dir)
        self.setup_schema()

    def connect(self, data_dir):
        db = sqlite3.connect(join(data_dir, DataAccess.DB_FILE_NAME),
                             check_same_thread=False,
                             detect_types=sqlite3.PARSE_DECLTYPES)
        return db

    @staticmethod
    def get_sampling_rate(device_and_signal_name):
        rates_dict = {  # In Hz
            "cns_ap-mean": 2.5,
            "cns_ap-syst": 2.5,
            "cns_ap-dias": 2.5,
            "cns_ap-na": 2.5,
            "cns_ci-na": 2.5,
            "cns_co-na": 2.5,
            "cns_cpi-na": 2.5,
            "cns_cpo-na": 2.5,
            "cns_dpmx-na": 2.5,
            "cns_elwi-na": 2.5,
            "cns_evlw-na": 2.5,
            "cns_gedi-na": 2.5,
            "cns_gedv-na": 2.5,
            "cns_gef-na": 2.5,
            "cns_itbi-na": 2.5,
            "cns_itbv-na": 2.5,
            "cns_pcco-na": 2.5,
            "cns_ppv-na": 2.5,
            "cns_pvpi-na": 2.5,
            "cns_sv-na": 2.5,
            "cns_svi-na": 2.5,
            "cns_svr-na": 2.5,
            "cns_svv-na": 2.5,
            "cns_art-mean": 1,
            "cns_art-syst": 1,
            "cns_art-dias": 1,
            "cns_pbto2-na": 0.5,
            "cns_ict-na": 0.5,
            "cns_cpp-na": 1,
            "cns_cpp2-na": 1,
            "cns_cvp-mean": 1,
            "cns_icp-mean": 1,
            "cns_icp2-mean": 1,
            "cns_hr-na": 1,
            "cns_lap-mean": 1,
            "cns_rr-na": 1,
            "cns_spo2-na": 1,
            "draeger_art": 100,
            "draeger_cvp": 100,
            "draeger_spo2": 100,
            "draeger_resp": 100,
            "draeger_icp": 100,
            "draeger_icp2": 100,
            "cns_cstat-na": 1,
            "cns_etco2-na": 1,
            "cns_expminvol-na": 1,
            "cns_fio2-na": 1,
            "cns_ftotal-na": 1,
            "cns_peep-na": 1,
            "cns_pinsp-na": 1,
            "cns_pmean-na": 1,
            "cns_pminimum-na": 1,
            "cns_ppeak-na": 1,
            "cns_pplateau-na": 1,
            "cns_rinsp-na": 1,
            "cns_rsb-na": 1,
            "cns_te-na": 1,
            "cns_ti-na": 1,
            "cns_fspontpct-na": 1,
            "cns_vte-na": 1,
            "cns_tinfinity-a": 1,
            "cns_tinfinity-b": 1,
            "cns_nbp-mean": 1,
        }
        if device_and_signal_name in rates_dict:
            return rates_dict[device_and_signal_name]
        else:
            return None

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

    def get_items_by_id_set(self, patient_id=None, id_set=set(), table_name="CHARTEVENTS",
                            get_subjects=False, limit=None, offset=None, value_case=None):
        if get_subjects:
            columns = "DISTINCT subject_id"
            subject_clause = ""
            order_clause = ""
        else:
            columns = "charttime, valuenum" \
                if value_case is None else \
                "charttime, {VALUE_CASE}".format(VALUE_CASE=value_case)
            subject_clause = "AND SUBJECT_ID = '{SUBJECT_ID}'".format(SUBJECT_ID=patient_id)
            order_clause = " ORDER BY charttime "

        if offset is None:
            offset_clause = ""
        else:
            offset_clause = "OFFSET {OFFSET_NUM}".format(OFFSET_NUM=offset)

        if limit is None:
            limit_clause = ""
        else:
            limit_clause = "LIMIT {LIMIT_NUM}".format(LIMIT_NUM=limit)

        error_clause = ""
        if table_name == "CHARTEVENTS":
            error_clause = "{TABLE_NAME}.error IS NOT 1 AND ".format(TABLE_NAME=table_name)

        id_set = map(str, id_set)
        id_set_string = ", ".join(id_set)

        query = ("SELECT {COLUMNS} "
                 "FROM {TABLE_NAME} "
                 "WHERE {ERROR_CLAUSE} ITEMID IN ({ID_SET}) "
                 "{SUBJECT_CLAUSE} {ORDER_CLAUSE} {LIMIT_CLAUSE} {OFFSET_CLAUSE};") \
            .format(COLUMNS=columns,
                    TABLE_NAME=table_name,
                    ID_SET=id_set_string,
                    ERROR_CLAUSE=error_clause,
                    SUBJECT_CLAUSE=subject_clause,
                    ORDER_CLAUSE=order_clause,
                    LIMIT_CLAUSE=limit_clause,
                    OFFSET_CLAUSE=offset_clause)
        result = self.db.execute(query).fetchall()

        if value_case is not None:
            return filter(lambda x: x[1] is not None, result)
        else:
            return result

    def get_spo2_values(self, patient_id):
        # From: https://github.com/MIT-LCP/mimic-code/blob/master/concepts/firstday/blood-gas-first-day-arterial.sql
        spo2_item_ids = {
            "646",
            "220277"
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=spo2_item_ids)

    def get_etco2_values(self, patient_id):
        etco2_item_ids = {
            "1817",
            "228640",
            "228641"
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=etco2_item_ids)

    def get_fio2_values(self, patient_id):
        # From: https://github.com/MIT-LCP/mimic-code/blob/master/concepts/firstday/blood-gas-first-day-arterial.sql
        fio2_item_ids = {
            "3420",    # FiO2
            "190",     # FiO2 set
            "223835",  # Inspired O2 Fraction (FiO2)
            "3422"     # FiO2 [measured]
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=fio2_item_ids,
                                        value_case="case"
                                                   "    when itemid = 223835 "
                                                   "      then case "
                                                   "        when valuenum > 0 and valuenum <= 1 "
                                                   "          then valuenum * 100 "
                                                   "        when valuenum > 1 and valuenum < 21 "
                                                   "          then null "
                                                   "        when valuenum >= 21 and valuenum <= 100 "
                                                   "          then valuenum "
                                                   "        else null end "
                                                   "  when itemid in (3420, 3422) "
                                                   "      then valuenum "
                                                   "  when itemid = 190 and valuenum > 0.20 and valuenum < 1 "
                                                   "      then valuenum * 100 "
                                                   "else null end as valuenum ")

    def get_peep_values(self, patient_id):
        # From: https://github.com/MIT-LCP/mimic-code/blob/master/concepts/durations/ventilation-durations.sql
        peep_item_ids = {"60", "437", "505", "506", "686", "220339", "224700"}
        return self.get_items_by_id_set(patient_id=patient_id, id_set=peep_item_ids)

    def get_hr_values(self, patient_id):
        item_ids = {
            211, 220045
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_sysbp_values(self, patient_id):
        item_ids = {
            51, 442, 455, 6701, 220179, 220050
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_diasbp_values(self, patient_id):
        item_ids = {
            8368, 8440, 8441, 8555, 220180, 220051
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_meanbp_values(self, patient_id):
        item_ids = {
            456, 52, 6702, 443, 220052, 220181, 225312
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_rr_values(self, patient_id):
        item_ids = {
            615, 618, 220210, 224690
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_tempc_values(self, patient_id):
        item_ids = {
            223762, 676,
            223761, 678
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids,
                                        value_case="case when itemid in (223761, 678) "
                                                   "then(valuenum - 32) / 1.8 else valuenum end as valuenum")

    def get_icp_values(self, patient_id):
        item_ids = {
            226, 220765
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_cpp_values(self, patient_id):
        item_ids = {
            227066,
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_tidal_volume_values(self, patient_id):
        item_ids = {
            639, 654, 681, 682, 683, 684, 224685, 224684, 224686
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_inspiratory_time_values(self, patient_id):
        item_ids = {
            224738, 2000
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_resistance_values(self, patient_id):
        item_ids = {
            220283
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_plateau_pressure_values(self, patient_id):
        item_ids = {
            543, 224696
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_min_pressure_values(self, patient_id):
        item_ids = {
            436, 3143, 6864, 6552, 7176
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_mean_pressure_values(self, patient_id):
        item_ids = {
            444, 224697
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_peak_pressure_values(self, patient_id):
        item_ids = {
            218, 535, 224695, 1686, 6047, 507
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_pinsp_values(self, patient_id):
        item_ids = {
            227187
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_ftotal_values(self, patient_id):
        item_ids = {
            619, 224688
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_pao2_values(self, patient_id):
        # From: https://github.com/MIT-LCP/mimic-code/blob/master/concepts/firstday/blood-gas-first-day.sql
        bloodgas_item_ids = {
            "50821",  # PaO2
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=bloodgas_item_ids, table_name="labevents")

    def get_fio2_lab_values(self, patient_id):
        # From: https://github.com/MIT-LCP/mimic-code/blob/master/concepts/firstday/blood-gas-first-day.sql
        bloodgas_item_ids = {
            "50816",  # FiO2
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=bloodgas_item_ids, table_name="labevents",
                                        value_case="case when valuenum < 20 then null "
                                                   " when valuenum > 100 then null "
                                                   " else valuenum end as valuenum")

    def get_ventilated_patients(self, offset=None, limit=None):
        ventilation_indicators_item_ids = {
            "720", "223849", "223848", "445", "448", "449", "450", "1340", "1486", "1600", "224687",
            "639", "654", "681", "682", "683", "684", "224685", "224684", "224686",
            "218", "436", "535", "444", "224697", "224695", "224696", "224746", "224747",
            "221", "1", "1211", "1655", "2000", "226873", "224738", "224419", "224750", "227187",
            "543", "5865", "5866", "224707", "224709", "224705",  "224706", "60", "437", "505", "506",
            "686", "220339", "224700", "3459", "501", "502", "503", "224702", "223",
            "667", "668", "669", "670", "671", "672", "224701"
        }

        with self.db:
            self.db.execute("CREATE TEMP VIEW first_admission_time AS "
                            " "
                            "  SELECT"
                            "   p.subject_id, "
                            "      MIN( ROUND( (cast(admittime as date) - cast(dob as date)), 2) )"
                            "          AS first_admit_age"
                            "  FROM patients p"
                            "  INNER JOIN admissions a"
                            "  ON p.subject_id = a.subject_id"
                            "  GROUP BY p.subject_id, p.dob, p.gender"
                            "  ORDER BY p.subject_id;")

        patient_query = (" SELECT"
                         "    subject_id"
                         " FROM first_admission_time "
                         " WHERE first_admit_age >= 14 AND first_admit_age < 89"
                         " ORDER BY subject_id")

        patients = map(lambda x: x[0], self.db.execute(patient_query).fetchall())

        matched_patients = set()
        for i, patient in enumerate(patients):
            matched = False
            for item_id in ventilation_indicators_item_ids:
                return_value = self.db.execute("SELECT EXISTS( SELECT 1 "
                                               "FROM chartevents "
                                               "WHERE chartevents.error IS NOT 1 AND "
                                               "ITEMID = {itemid} AND "
                                               "subject_id = {subject_id});"
                                               .format(itemid=item_id, subject_id=patient)).fetchone()
                if return_value[0] == 1:
                    matched = True
                    break
            if matched:
                matched_patients.add(patient)
                if limit is not None and len(matched_patients) == limit:
                    break

        return matched_patients
