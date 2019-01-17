#!/bin/bash

# The root folder for which to calculate the summary measures for a given model,
# e.g. /some/directory/pm_news16a7k_cfrnet_mse_1
# The root folder must contain the individual run data (./run_$i/run.txt)
FOLDER_PATH="$1"

export PYTHONPATH="./drnet/:$PYTHONPATH"
# The following command merges all run files for repeated runs (run_$i/run.txt)
# into a single summary file ($FOLDER_PATH/summary.txt).
python ./drnet/apps/main.py --dataset=./ --with_rnaseq --do_train --do_hyperopt --num_hyperopt_runs=10 --do_evaluate --fraction_of_data_set=1.00 --num_units=16 --num_layers=2 --seed=909 --num_epochs=100 --learning_rate=0.001 --dropout=0.0 --batch_size=4 --do_merge_lsf --l2_weight=0.000 --imbalance_loss_weight=0.0 --benchmark=jobs --method=nn --early_stopping_patience=7 --do_not_save_predictions --validation_set_fraction=0.24 --test_set_fraction=0.2 --with_propensity_batch --early_stopping_on_pehe --experiment_index=27 --num_treatments=2 --output_directory=$FOLDER_PATH > /dev/null 2>&1

wait

function join_by { local d=$1; shift; echo -n "$1"; shift; printf "%s" "${@/#/$d}"; }

i=0
mise=()
dpe=()
pe=()
yourfilenames=`ls $FOLDER_PATH | grep run`
for EACH_FILE in $yourfilenames
do
    result=$(cat $FOLDER_PATH/$EACH_FILE/*.txt | sed  -n '/^INFO: Best_test_score/,/^.*}/p' | paste -sd " " - | sed -e 's/\(.*\)array(.*)/\1None/' | sed 's/.*{\(.*\)}/{\1}/')

    if [[ ! "$result" == *} ]]; then
        result=$(cat $FOLDER_PATH/$EACH_FILE/*.txt | grep Best_test_score  | sed -e 's/\(.*\)array(.*)/\1None/' | sed 's/.*{\(.*\)}/{\1}/')
    fi

    if [ -z "$result" ]
    then
        echo "$FOLDER_PATH/$EACH_FILE was broken. Check the output run.txt file to ensure the command has finished."
    else
        # echo "print($result['rmise'])"
        mise[$i]=$(python -c "print($result['rmise'])")
        dpe[$i]=$(python -c "print($result['dpe'])")
        pe[$i]=$(python -c "print($result['pe'])")
        ((i++))
    fi
done

m1=$(python -c "import numpy as np; print(np.mean([$(join_by , "${mise[@]}")]))")
s1=$(python -c "import numpy as np; print(np.std([$(join_by , "${mise[@]}")]))")
m2=$(python -c "import numpy as np; print(np.mean([$(join_by , "${dpe[@]}")]))")
s2=$(python -c "import numpy as np; print(np.std([$(join_by , "${dpe[@]}")]))")
m3=$(python -c "import numpy as np; print(np.mean([$(join_by , "${pe[@]}")]))")
s3=$(python -c "import numpy as np; print(np.std([$(join_by , "${pe[@]}")]))")

m1=$(printf "%.1f" $m1)
s1=$(printf "%.1f" $s1)
m2=$(printf "%.1f" $m2)
s2=$(printf "%.1f" $s2)
m3=$(printf "%.1f" $m3)
s3=$(printf "%.1f" $s3)

echo "{$m1} \$\\pm\$ $s1 & {$m2} \$\\pm\$ $s2 & {$m3} \$\\pm\$ $s3"
