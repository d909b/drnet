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

import os
import sys
import numpy as np

BATCH_NUMBER = "1"

default_params = "--dataset={DATASET_PATH} " \
                 "--with_rnaseq " \
                 "--do_train " \
                 "--do_evaluate " \
                 "--num_hyperopt_runs={NUM_HYPEROPT_RUNS} " \
                 "--do_hyperopt " \
                 "--fraction_of_data_set=1.00 " \
                 "--num_units=64 " \
                 "--num_layers=3 " \
                 "--seed={i} " \
                 "--num_epochs={NUM_EPOCHS} " \
                 "--learning_rate=0.001 " \
                 "--dropout=0.0 " \
                 "--batch_size=50 " \
                 "--output_directory={OUTPUT_FOLDER}/{NAME}/run_{i} " \
                 "--l2_weight=0.0 " \
                 "--imbalance_loss_weight=0.0 " \
                 "--benchmark={DATASET} " \
                 "--num_treatments={NUM_TREATMENTS} " \
                 "--strength_of_assignment_bias={KAPPA} " \
                 "--tcga_num_features={TCGA_FEATURES} " \
                 "--early_stopping_patience={EARLY_STOPPING_PATIENCE} " \
                 "--do_not_save_predictions " \
                 "--propensity_batch_probability={PBM_PROBABILITY} " \
                 "--experiment_index={i} " \
                 "--num_randomised_neighbours=1 " \
                 "--num_exposure_strata={NUM_EXPOSURE_STRATA} "

command_params_pbm = "--method={MODEL_TYPE} " \
                     "--with_propensity_batch " \
                     "--imbalance_loss_weight=0.0 "

command_params_pbm_no_tarnet = "--method={MODEL_TYPE} " \
                               "--with_propensity_batch " \
                               "--imbalance_loss_weight=0.0 " \
                               "--match_on_covariates " \
                               "--do_not_use_tarnet "

command_params_pbm_no_repeat = "--method={MODEL_TYPE} " \
                               "--with_propensity_batch " \
                               "--imbalance_loss_weight=0.0 " \
                               "--match_on_covariates " \
                               "--do_not_use_multiple_exposure_inputs "

command_params_pbm_no_strata = "--method={MODEL_TYPE} " \
                               "--with_propensity_batch " \
                               "--imbalance_loss_weight=0.0 " \
                               "--match_on_covariates " \
                               "--do_not_use_exposure_strata "

command_params_no_tarnet = "--method=nn+ " \
                           "--imbalance_loss_weight=0.0 " \
                           "--match_on_covariates " \
                           "--do_not_use_tarnet "

command_params_tarnet_no_repeat = "--method=nn+ " \
                                  "--imbalance_loss_weight=0.0 " \
                                  "--match_on_covariates " \
                                  "--do_not_use_multiple_exposure_inputs "

command_params_tarnet_no_strata = "--method=nn+ " \
                                  "--imbalance_loss_weight=0.0 " \
                                  "--match_on_covariates " \
                                  "--do_not_use_exposure_strata "

command_params_pbm_mahalanobis = "--method={MODEL_TYPE} " \
                                 "--with_propensity_batch " \
                                 "--imbalance_loss_weight=0.0 " \
                                 "--match_on_covariates "

command_params_psm = "--method=psm " \
                     "--imbalance_loss_weight=0.0 "

command_params_psmpbm = "--method=psmpbm " \
                        "--imbalance_loss_weight=0.0 "

command_params_psmpbm_mahal = "--method=psmpbm " \
                              "--imbalance_loss_weight=0.0 " \
                              "--match_on_covariates "

command_params_ganite = "--method=ganite " \
                        "--imbalance_loss_weight=0.0 "

command_params_tarnet = "--method=nn+ " \
                        "--imbalance_loss_weight=0.0 "

command_params_cfrnet = "--method=nn+ " \
                        "--imbalance_loss_weight=1.0 "

command_params_cf = "--method=cf " \
                    "--imbalance_loss_weight=0.0 "

command_params_rf = "--method=rf " \
                    "--imbalance_loss_weight=0.0 "

command_params_bart = "--method=bart " \
                      "--imbalance_loss_weight=0.0 "

command_params_knn = "--method=knn " \
                     "--imbalance_loss_weight=0.0 "

command_params_gps = "--method=gps " \
                     "--imbalance_loss_weight=0.0 "

command_params_ols1 = "--method=ols1 " \
                      "--imbalance_loss_weight=0.0 "

command_params_ols2 = "--method=ols2 " \
                      "--imbalance_loss_weight=0.0 "

command_params_tarnetpd = "--method=nn+ " \
                          "--with_propensity_dropout " \
                          "--imbalance_loss_weight=0.0 "

command_params_mse = " "
command_params_pehe = "--early_stopping_on_pehe "

command_params_exposure = "--with_exposure " \
                          "--model_selection_metric=nn_rmise "

command_template = "mkdir -p {OUTPUT_FOLDER}/{NAME}/run_{i}/ && " \
                   "CUDA_VISIBLE_DEVICES='' {SUB_COMMAND} "

ALL_MODELS = [
    "pbm_mahal", "no_tarnet", "tarnet_no_repeat", "tarnet_no_strata",
    "knn", "psmpbm_mahal", "gps", "bart", "cf", "ganite", "tarnetpd", "tarnet", "cfrnet"
]


DEFAULT_NUM_REPEATS = 5
DEFAULT_EXPOSURE_STRATA = 5


def model_is_pbm_variant(model_type):
    return model_type.startswith("pbm")


def get_cnews_config():
    num_tcga_features = None
    num_exposure_stratas = None
    num_hyperopt_runs = 10
    num_epochs = 100
    early_stopping_patience = 30
    num_repeats = DEFAULT_NUM_REPEATS
    treatment_set = [2, 4, 8, 16]
    kappa_set = [10, 10, 10, 7]
    model_set = ALL_MODELS
    es_set = ["mse"]*len(model_set)
    pbm_percentages = [1.0]*len(es_set)
    return num_hyperopt_runs, num_epochs, early_stopping_patience, num_repeats, treatment_set, \
           kappa_set, model_set, es_set, pbm_percentages, num_tcga_features, num_exposure_stratas


def get_cnews_kappa_config():
    num_tcga_features = None
    num_exposure_stratas = None
    num_hyperopt_runs = 10
    num_epochs = 100
    early_stopping_patience = 30
    num_repeats = 1
    treatment_set = [2, 2, 2, 2, 2, 2, 2]
    kappa_set = [5, 7, 10, 12, 15, 17, 20]
    model_set = ["tarnet", "tarnet_no_strata", "no_tarnet", "gps"]
    es_set = ["mse"]*len(model_set)
    pbm_percentages = [1.0]*len(es_set)
    return num_hyperopt_runs, num_epochs, early_stopping_patience, num_repeats, treatment_set, \
           kappa_set, model_set, es_set, pbm_percentages, num_tcga_features, num_exposure_stratas


def get_tcga_confounding_config():
    num_exposure_stratas = None
    num_hyperopt_runs = 5
    num_epochs = 200
    early_stopping_patience = 30
    num_repeats = 1
    max_tcga_features = 20531
    num_tcga_features = np.rint(np.array([0.2, 0.5]) * max_tcga_features).tolist()
    treatment_set = [3] * len(num_tcga_features)
    kappa_set = [10] * len(num_tcga_features)
    model_set = ["tarnet"]
    es_set = ["mse"]*len(model_set)
    pbm_percentages = [1.0]*len(es_set)
    return num_hyperopt_runs, num_epochs, early_stopping_patience, num_repeats, treatment_set, \
           kappa_set, model_set, es_set, pbm_percentages, num_tcga_features, num_exposure_stratas


def get_tcga_config():
    num_tcga_features = None
    num_exposure_stratas = None
    num_hyperopt_runs = 5
    num_epochs = 200
    early_stopping_patience = 30
    num_repeats = DEFAULT_NUM_REPEATS
    treatment_set = [3]
    kappa_set = [10]
    # "knn" and "bart" are too slow for this number of features.
    model_set = set(ALL_MODELS) - {"knn", "bart"}
    es_set = ["mse"]*len(model_set)
    pbm_percentages = [1.0]*len(es_set)
    return num_hyperopt_runs, num_epochs, early_stopping_patience, num_repeats, treatment_set, \
           kappa_set, model_set, es_set, pbm_percentages, num_tcga_features, num_exposure_stratas


def get_icu_exposure_config():
    num_tcga_features = None
    num_exposure_stratas = [2, 4, 6, 8, 10]
    num_hyperopt_runs = 10
    num_epochs = 300
    early_stopping_patience = 30
    num_repeats = 1
    treatment_set = [3]
    kappa_set = [10]
    model_set = ["tarnet"]*len(num_exposure_stratas)
    es_set = ["mse"]*len(model_set)
    pbm_percentages = [1.0]*len(es_set)
    return num_hyperopt_runs, num_epochs, early_stopping_patience, num_repeats, treatment_set, \
           kappa_set, model_set, es_set, pbm_percentages, num_tcga_features, num_exposure_stratas


def get_icu_config():
    num_tcga_features = None
    num_exposure_stratas = None
    num_hyperopt_runs = 10
    num_epochs = 300
    early_stopping_patience = 30
    num_repeats = DEFAULT_NUM_REPEATS
    treatment_set = [3]
    kappa_set = [10]
    model_set = ALL_MODELS
    es_set = ["mse"]*len(model_set)
    pbm_percentages = [1.0]*len(es_set)
    return num_hyperopt_runs, num_epochs, early_stopping_patience, num_repeats, treatment_set, \
           kappa_set, model_set, es_set, pbm_percentages, num_tcga_features, num_exposure_stratas


def get_dataset_params(DATASET, is_exposure=True):
    exposures, num_exposure_stratas, num_tcga_features = None, None, None
    if DATASET == "news_treatment_assignment_bias":
        # Kappa t2
        num_hyperopt_runs, num_epochs, early_stopping_patience, num_repeats, treatment_set, \
        kappa_set, model_set, es_set, pbm_percentages, num_tcga_features, num_exposure_stratas \
            = get_cnews_kappa_config()
        DATASET = "news"
    elif DATASET == "news":
        # Default
        num_hyperopt_runs, num_epochs, early_stopping_patience, num_repeats, treatment_set, \
        kappa_set, model_set, es_set, pbm_percentages, num_tcga_features, num_exposure_stratas \
            = get_cnews_config()
    elif DATASET == "icu_exposure":
        # Exposure strata
        num_hyperopt_runs, num_epochs, early_stopping_patience, num_repeats, treatment_set, \
        kappa_set, model_set, es_set, pbm_percentages, num_tcga_features, num_exposure_stratas \
            = get_icu_exposure_config()
        DATASET = "icu"
    elif DATASET == "icu":
        num_hyperopt_runs, num_epochs, early_stopping_patience, num_repeats, treatment_set, \
        kappa_set, model_set, es_set, pbm_percentages, num_tcga_features, num_exposure_stratas \
            = get_icu_config()
    elif DATASET == "tcga_confounding":
        # Confounding
        num_hyperopt_runs, num_epochs, early_stopping_patience, num_repeats, treatment_set, \
        kappa_set, model_set, es_set, pbm_percentages, num_tcga_features, num_exposure_stratas \
            = get_tcga_confounding_config()
        DATASET = "tcga"
    elif DATASET == "tcga":
        # Default
        num_hyperopt_runs, num_epochs, early_stopping_patience, num_repeats, treatment_set, \
        kappa_set, model_set, es_set, pbm_percentages, num_tcga_features, num_exposure_stratas \
            = get_tcga_config()

    if num_tcga_features is None:
        num_tcga_features = [0]*len(kappa_set)

    if exposures is None:
        exposures = [is_exposure]*len(kappa_set)

    if num_exposure_stratas is None:
        num_exposure_stratas = [DEFAULT_EXPOSURE_STRATA]*len(model_set)

    return DATASET, num_hyperopt_runs, num_epochs, early_stopping_patience, num_repeats, treatment_set, kappa_set, \
           model_set, es_set, pbm_percentages, num_tcga_features, exposures, num_exposure_stratas


def run(DATASET, DATASET_PATH, OUTPUT_FOLDER, SUB_COMMAND, LOG_FILE):
    DATASET, num_hyperopt_runs, num_epochs, early_stopping_patience, num_repeats, \
    treatment_set, kappa_set, model_set, es_set, pbm_percentages, num_tcga_features, exposures, num_exposure_stratas \
        = get_dataset_params(DATASET)

    for num_treatments, kappa, tcga_features, exposure in zip(treatment_set, kappa_set, num_tcga_features, exposures):
        for model_type, early_stopping_type, pbm_percentage, num_exposure_strata in \
                zip(model_set, es_set, pbm_percentages, num_exposure_stratas):
            if model_type == "pbm":
                command_params = command_params_pbm
            elif model_type == "ganite":
                command_params = command_params_ganite
            elif model_type == "pbm_mahal":
                command_params = command_params_pbm_mahalanobis
            elif model_type == "pbm_mahal_no_tarnet":
                command_params = command_params_pbm_no_tarnet
            elif model_type == "pbm_mahal_no_repeat":
                command_params = command_params_pbm_no_repeat
            elif model_type == "pbm_mahal_no_strata":
                command_params = command_params_pbm_no_strata
            elif model_type == "no_tarnet":
                command_params = command_params_no_tarnet
            elif model_type == "tarnet_no_repeat":
                command_params = command_params_tarnet_no_repeat
            elif model_type == "tarnet_no_strata":
                command_params = command_params_tarnet_no_strata
            elif model_type == "psm":
                command_params = command_params_psm
            elif model_type == "psmpbm":
                command_params = command_params_psmpbm
            elif model_type == "psmpbm_mahal":
                command_params = command_params_psmpbm_mahal
            elif model_type == "tarnet":
                command_params = command_params_tarnet
            elif model_type == "tarnetpd":
                command_params = command_params_tarnetpd
            elif model_type == "cfrnet":
                command_params = command_params_cfrnet
            elif model_type == "cf":
                command_params = command_params_cf
            elif model_type == "rf":
                command_params = command_params_rf
            elif model_type == "bart":
                command_params = command_params_bart
            elif model_type == "knn":
                command_params = command_params_knn
            elif model_type == "gps":
                command_params = command_params_gps
            elif model_type == "ols1":
                command_params = command_params_ols1
            elif model_type == "ols2":
                command_params = command_params_ols2
            else:
                command_params = command_params_tarnet

            if model_is_pbm_variant(model_type):
                command_params = command_params.format(MODEL_TYPE="nn+")

            if early_stopping_type == "pehe":
                command_early_stopping = command_params_pehe
            else:
                command_early_stopping = command_params_mse

            extra_commands = ""
            if exposure:
                extra_commands += command_params_exposure

            name = "drnet_{DATASET}{NUM_TREATMENTS}a{KAPPA}k{EXP_STRATA}{PBM_P}{TCGA}_{MODEL_TYPE}_{EARLY_STOPPING_TYPE}_{BATCH_NUMBER}" \
                .format(DATASET=DATASET,
                        KAPPA=kappa,
                        PBM_P="{0:.2f}".format(pbm_percentage) + "p" if pbm_percentage != 1.0 else "",
                        TCGA="{0:d}".format(int(tcga_features)) + "f" if tcga_features != 0 else "",
                        NUM_TREATMENTS=num_treatments,
                        BATCH_NUMBER=BATCH_NUMBER,
                        MODEL_TYPE=model_type,
                        EARLY_STOPPING_TYPE=early_stopping_type,
                        EXP_STRATA="{0:d}".format(int(num_exposure_strata)) + "e"
                                   if num_exposure_strata != DEFAULT_EXPOSURE_STRATA else "")

            for i in range(0, num_repeats):
                local_log_file = LOG_FILE.format(NAME=name, i=i)

                print((command_template + default_params + command_params + command_early_stopping + extra_commands
                       + "&> {LOG_FILE}")
                      .format(SUB_COMMAND=SUB_COMMAND,
                              LOG_FILE=local_log_file,
                              NAME=name,
                              DATASET=DATASET,
                              DATASET_PATH=DATASET_PATH,
                              OUTPUT_FOLDER=OUTPUT_FOLDER,
                              KAPPA=kappa,
                              TCGA_FEATURES=int(tcga_features),
                              NUM_TREATMENTS=num_treatments,
                              NUM_HYPEROPT_RUNS=num_hyperopt_runs,
                              NUM_EPOCHS=num_epochs,
                              PBM_PROBABILITY=pbm_percentage,
                              NUM_EXPOSURE_STRATA=num_exposure_strata,
                              EARLY_STOPPING_PATIENCE=early_stopping_patience,
                              i=i))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("USAGE: ./run_all_experiments.py {PATH_TO_FOLDER_CONTAINING_MAIN.PY} {DATASET_NAME} {DATABASE_PATH} {OUTPUT_FOLDER}\n"
              "       e.g. ./run_all_experiments.py ./ news ./data ./results\n"
              "       where \n"
              "         PATH_TO_FOLDER_CONTAINING_MAIN.PY is the path to the directory that contains main.py\n"
              "         DATASET_NAME is one of (news, tcga, icu)\n"
              "         DATABASE_PATH is the path to the directory containing tcga.db and news.db \n"
              "                       (See README.md on where to download tcga.db and news.db)\n"
              "         OUTPUT_FOLDER is the path to the directory to which you want to save experiment results.\n",
              file=sys.stderr)
    else:
        # Path where the python executable file is located.
        BINARY_FOLDER = sys.argv[1]

        # Dataset to use. One of: (news, tcga, icu).
        DATASET = sys.argv[2]

        # Path where the SQLite databases for each dataset (tcga.db, news.db, icu_windows.db) are located.
        DATASET_PATH = sys.argv[3]

        # Folder to write output files and intermediary models to.
        OUTPUT_FOLDER = sys.argv[4]

        # Python command to execute.
        SUB_COMMAND = "python {BINARY}".format(BINARY=os.path.join(BINARY_FOLDER, "main.py"))

        # Folder to write the log file to.
        # Do not change if you want to use the run_results.sh script for result parsing.
        LOG_FILE = os.path.join(OUTPUT_FOLDER, "{NAME}/run_{i}/run.txt")

        run(DATASET, DATASET_PATH, OUTPUT_FOLDER, SUB_COMMAND, LOG_FILE)
