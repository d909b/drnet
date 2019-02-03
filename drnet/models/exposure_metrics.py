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

import matplotlib
import numpy as np
matplotlib.rcParams.update({'font.size': 16})
matplotlib.use('Agg')
from os.path import join
import matplotlib.pyplot as plt
from drnet.data_access.mahalanobis_batch import MahalanobisBatch


def calculate_exposure_metrics(model, benchmark, train_ids, all_x, all_z, y_true_f, num_treatments,
                               samples_power_of_two=6,
                               reduce_dimensionality=True,
                               dimensionality_reduction_limit=8,
                               dimensionality_reduction_target=8,
                               plot_limit=100,
                               do_plot=False,
                               output_directory=""
                               ):
    from scipy.integrate import romb
    from scipy.optimize import minimize

    has_z = len(all_x) == len(all_z)
    if has_z:
        values = zip(all_x, all_z)
    else:
        values = all_x

    # Pass only factual outcomes for preparing the matching lists.
    benchmark.set_assign_counterfactuals(False)
    plot_count = 0

    # Prepare matching.
    matching = MahalanobisBatch()
    matching.make_propensity_lists(train_ids, benchmark,
                                   reduce_dimensionality=reduce_dimensionality,
                                   dimensionality_reduction_limit=dimensionality_reduction_limit,
                                   dimensionality_reduction_target=dimensionality_reduction_target,
                                   with_exposure=True)
    id_map = dict(zip(train_ids, np.arange(len(train_ids))))

    benchmark.set_assign_counterfactuals(True)

    num_integration_samples = 2**samples_power_of_two + 1
    step_size = 1./num_integration_samples

    average_treatment_curves_true = [None for _ in range(num_treatments)]
    average_treatment_curves_pred = [None for _ in range(num_treatments)]
    mises, nn_mises, policy_errors, dosage_policy_errors, pred_best, pred_vals, true_best = [], [], [], [], [], [], []
    for value in values:
        if has_z:
            x, z = value
        else:
            x, z = value, value

        for treatment_idx in range(num_treatments):
            true_dose_response_curve = benchmark.get_dose_response_curve(z, treatment_idx)
            treatment_strengths = np.linspace(np.finfo(float).eps, 1, num_integration_samples)
            true_dose_response = np.array([true_dose_response_curve(ts) for ts in treatment_strengths])
            pred_dose_response = model.predict([np.repeat(np.expand_dims(x, axis=0), num_integration_samples, axis=0),
                                                np.repeat(treatment_idx, num_integration_samples),
                                                treatment_strengths])
            true_dose_response *= benchmark.get_scaling_constant()

            if isinstance(pred_dose_response, list):
                pred_dose_response = pred_dose_response[-1]

            def nn_dose_response_curve(s):
                nn_x, nn_id = matching.get_closest_in_propensity_lists(x, treatment_idx, 1, treatment_strength=s)
                nn_y = y_true_f[id_map[nn_id]]
                return nn_y
            nn_dose_response = np.array([nn_dose_response_curve(ts) for ts in treatment_strengths])

            if average_treatment_curves_true[treatment_idx] is None:
                average_treatment_curves_true[treatment_idx] = true_dose_response
            else:
                average_treatment_curves_true[treatment_idx] += true_dose_response

            if average_treatment_curves_pred[treatment_idx] is None:
                average_treatment_curves_pred[treatment_idx] = pred_dose_response
            else:
                average_treatment_curves_pred[treatment_idx] += pred_dose_response

            def pred_dose_response_curve(s):
                ret_val = model.predict([np.expand_dims(x, axis=0),
                                         np.expand_dims(treatment_idx, axis=0),
                                         np.expand_dims(s, axis=0)])
                if isinstance(ret_val, list):
                    ret_val = ret_val[-1]
                return ret_val

            if pred_dose_response.ndim == 2:
                pred_dose_response = pred_dose_response[:, 0]

            # Integrate the squared difference in the real and predicted dosage response curves.
            mise = romb(np.square(true_dose_response - pred_dose_response),
                        dx=step_size)
            mises.append(mise)

            nn_mise = romb(np.square(nn_dose_response - pred_dose_response),
                           dx=step_size)
            nn_mises.append(nn_mise)

            best_encountered_x = treatment_strengths[np.argmax(pred_dose_response)]
            min_true = minimize(lambda x: -1*true_dose_response_curve(x),
                                x0=[0.5], method="SLSQP", bounds=[(0, 1)]).fun * -1 * benchmark.get_scaling_constant()
            min_pred = minimize(lambda x: -1*pred_dose_response_curve(x),
                                x0=[best_encountered_x], method="SLSQP", bounds=[(0, 1)])
            min_pred_x, min_pred_val = min_pred.x[0], -1*min_pred.fun
            min_pred = true_dose_response_curve(min_pred_x) * benchmark.get_scaling_constant()
            dosage_policy_error = (min_true - min_pred) ** 2

            # Plot dosage-response curves.
            if do_plot and plot_count < plot_limit:
                plt.clf()

                plt.title("MISE = " + str(np.sqrt(mise)) + ", DPE" + str(np.sqrt(dosage_policy_error)))
                plt.plot(true_dose_response, c='#F59799')
                plt.plot(pred_dose_response, c='#9DC7EA')

                plt.savefig(join(output_directory, "img_" + str(plot_count) + ".pdf"))
                plot_count += 1

            dosage_policy_errors.append(dosage_policy_error)
            pred_best.append(min_pred)
            pred_vals.append(min_pred_val)
            true_best.append(min_true)
        selected_t_pred = np.argmax(pred_vals[-num_treatments:])
        selected_val = pred_best[-num_treatments:][selected_t_pred]
        selected_t_optimal = np.argmax(true_best[-num_treatments:])
        optimal_val = true_best[-num_treatments:][selected_t_optimal]
        policy_error = (optimal_val - selected_val)**2
        policy_errors.append(policy_error)

    num_samples, amises = len(all_x), []
    for treatment_idx in range(num_treatments):
        average_treatment_curves_true[treatment_idx] /= num_samples
        average_treatment_curves_pred[treatment_idx] /= num_samples
        amise = romb(np.square(average_treatment_curves_true[treatment_idx] -
                               average_treatment_curves_pred[treatment_idx]),
                     dx=step_size)
        amises.append(amise)

    return {
        "mise": np.mean(mises),
        "mise_std": np.std(mises),
        "rmise": np.sqrt(np.mean(mises)),
        "rmise_std": np.sqrt(np.std(mises)),
        "nn_rmise": np.sqrt(np.mean(nn_mises)),
        "nn_rmise_std": np.sqrt(np.std(nn_mises)),
        "pe": np.sqrt(np.mean(policy_errors)),
        "pe_std": np.sqrt(np.std(policy_errors)),
        "dpe": np.sqrt(np.mean(dosage_policy_errors)),
        "dpe_std": np.sqrt(np.std(dosage_policy_errors)),
        "amise": np.sqrt(amises),
        "aamise": np.sqrt(np.mean(amises)),
        "aamise_std": np.sqrt(np.std(amises)),
    }