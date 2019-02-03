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
from datetime import datetime
from bisect import bisect_right


def timestamp_string_to_datetime(timestamp):
    try:
        return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
    except:
        # Fallback in case microseconds are missing.
        return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")


def timestamp_from_string(timestamp_string):
    return int(timestamp_string_to_datetime(timestamp_string)
               .strftime("%s")) * 1000


def insert_patients(data_access, per_patient_data, output_directory, blood_gas_key, window_length_in_seconds,
                    db_origin="uzh"):
    from drnet.data_access.icu.data_access import DataAccess as OutputDataAccess
    global WINDOW_OFFSET
    global WINDOWS_TOTAL

    windows_data_access = OutputDataAccess(output_directory, is_factual=False)

    all_windows, all_patients, all_outcomes = [], [], []
    for patient_id in per_patient_data.keys():
        windows, blood_gas_values = per_patient_data[patient_id]

        for window, (blood_gas_timestamp, blood_gas_value) in zip(windows, blood_gas_values):
            for signal_name in window.keys():
                sampling_rate = data_access.get_sampling_rate(signal_name)

                if window[signal_name] is None:
                    vals = None
                    timestamps = None
                else:
                    vals = window[signal_name][:, 1]
                    timestamps = window[signal_name][:, 0]

                all_windows.append((WINDOW_OFFSET, signal_name,
                                    timestamps, vals,
                                    sampling_rate, window_length_in_seconds,
                                    db_origin))
                all_patients.append((patient_id, WINDOW_OFFSET))
            all_outcomes.append((WINDOW_OFFSET, blood_gas_key,
                                 blood_gas_timestamp, blood_gas_value,
                                 "mmHg"))
            WINDOW_OFFSET += 1

        with windows_data_access.db:
            windows_data_access.insert_many_values(OutputDataAccess.TABLE_OUTCOMES, all_outcomes)
            windows_data_access.insert_many_values(OutputDataAccess.TABLE_WINDOWS, all_windows)
            windows_data_access.insert_many_values(OutputDataAccess.TABLE_PATIENTS, all_patients)

        WINDOWS_TOTAL += len(all_outcomes)
        print("[INFO]: Added", len(all_patients) // len(window.keys()) // len(all_outcomes),
              "patients with", len(all_outcomes), "windows (total=", WINDOWS_TOTAL, ").",
              file=sys.stderr)

        all_windows, all_patients, all_outcomes = [], [], []


def get_windows_and_labels_mimic3(data_access, output_directory, patients, required_signals,
                                  window_length_in_seconds=60*60,
                                  normalise=False,
                                  blood_gas_key="pO2(a)/FIO2"):
    print("[INFO]: Generating data points and labels (mimic).", file=sys.stderr)

    all_signals = {
        "cns_spo2-na": data_access.get_spo2_values,
        "cns_etco2-na": data_access.get_etco2_values,
        "cns_fio2-na": data_access.get_fio2_values,
        "cns_peep-na": data_access.get_peep_values,
        "cns_art-mean": data_access.get_meanbp_values,
        "cns_art-sys": data_access.get_sysbp_values,
        "cns_art-dias": data_access.get_diasbp_values,
        "cns_rr-na": data_access.get_rr_values,
        "cns_hr-na": data_access.get_hr_values,
        "cns_tinfinity-a": data_access.get_tempc_values,
        "cns_icp-mean": data_access.get_icp_values,
        "cns_cpp-na": data_access.get_cpp_values,
        "cns_vte-na": data_access.get_tidal_volume_values,
        "cns_ti-na": data_access.get_inspiratory_time_values,
        "cns_rinsp-na": data_access.get_resistance_values,
        "cns_pplateau-na": data_access.get_plateau_pressure_values,
        "cns_ppeak-na": data_access.get_peak_pressure_values,
        "cns_pminimum-na": data_access.get_min_pressure_values,
        "cns_pmean-na": data_access.get_mean_pressure_values,
        "cns_pinsp-na": data_access.get_pinsp_values,
        "cns_ftotal-na": data_access.get_ftotal_values,
    }

    # Get blood draws.
    # For each blood draw get PEEP set, FIO2, etco2, spo2, and Temperature values.
    per_patient_data = {}
    for i, patient_id in enumerate(patients):
        blood_gas = data_access.get_pao2_values(patient_id)
        lab_fio2 = data_access.get_fio2_lab_values(patient_id)
        fio2_timestamps = map(lambda x: timestamp_from_string(x[0]), lab_fio2)

        if len(blood_gas) == 0 or len(lab_fio2) == 0:
            continue

        all_x, all_y = [], []
        for timestamp, blood_gas_value in blood_gas:
            timestamp = timestamp_from_string(timestamp)

            fio2_end_index = bisect_right(fio2_timestamps, timestamp)
            fio2_start_index = bisect_right(fio2_timestamps, timestamp - window_length_in_seconds)
            if fio2_end_index == 0 or fio2_end_index == fio2_start_index:
                continue  # No fio2 value before the pao2 value present.

            try:
                fio2_value = lab_fio2[fio2_end_index - 1][1] / 100.
                adjusted_blood_gas_value = blood_gas_value / fio2_value
            except:
                continue  # Not a valid fio2 or pao2 value (ex. type exception).

            windows, did_find_any = {}, False
            for signal_name, signal_data_fn in all_signals.items():
                signal_data = signal_data_fn(patient_id)
                signal_data = np.column_stack((map(lambda x: timestamp_from_string(x[0]), signal_data),
                                               map(lambda x: x[1], signal_data)))

                # Get closest in each signal.
                timestamps = signal_data[:, 0]
                end_idx = bisect_right(timestamps, timestamp)
                start_idx = bisect_right(timestamps, timestamp - window_length_in_seconds*1000)

                # Save to DB.
                did_find_result = len(signal_data) != 0 and start_idx != 0 and end_idx != start_idx
                if did_find_result:
                    did_find_any = True
                    assert len(signal_data[start_idx:end_idx]) != 0
                    windows[signal_name] = signal_data[start_idx:end_idx]
                else:
                    windows[signal_name] = None

                if not did_find_result and signal_name in required_signals:
                    did_find_any = False
                    break

            if did_find_any:
                all_x.append(windows)
                all_y.append((timestamp, adjusted_blood_gas_value))

        if len(all_x) != 0:
            per_patient_data[patient_id] = all_x, all_y
            insert_patients(data_access, per_patient_data, output_directory, blood_gas_key, window_length_in_seconds,
                            db_origin="mimic3")
            per_patient_data = {}

        print(i, "of", len(patients), file=sys.stderr)


def run(mimic3_database_directory, output_directory):
    required_signals = {
        "cns_fio2-na", "cns_peep-na",
        "cns_ftotal-na"
    }
    from drnet.data_access.icu.mimic3.data_access import DataAccess as Mimic3DataAccess

    da = Mimic3DataAccess(mimic3_database_directory)
    vps = da.get_ventilated_patients()
    get_windows_and_labels_mimic3(da, output_directory, vps, required_signals)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("USAGE: ./load_db_icu.py {DATABASE_PATH} {OUTPUT_FOLDER}\n"
              "       e.g. ./load_db_icu.py ./mimic3 ./data\n"
              "       where \n"
              "         DATABASE_PATH is the path to the directory containing your MIMIC3 DB in SQLite format \n"
              "                       (See README.md on how to obtain MIMIC3)\n"
              "         OUTPUT_FOLDER is the path to the directory to which you want to save the benchmark DB.\n",
              file=sys.stderr)
    else:
        mimic3_database_directory = sys.argv[1]
        output_directory = sys.argv[2]

        run(mimic3_database_directory, output_directory)
