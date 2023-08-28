#!/bin/sh

set -e

. ./scripts/activate.sh

VERSION="v1"
INSTANCES="1500000:1500100"
TOL="2.5"

# For debugging, the following options are useful.
# FLAGS="--smoketest"
# INSTANCES="1500000:1500008"
# TOL="5"

for size in 100
do
    python \
        scripts/train_nearest_neighbour.py \
        --path workspace/$VERSION/$size \
        $FLAGS

    python \
        scripts/train_random_forest.py \
        --path workspace/$VERSION/$size \
        $FLAGS

    python \
        scripts/train_single_sampling_network.py \
        --path workspace/$VERSION/$size \
        $FLAGS

    python \
        scripts/train_double_sampling_network.py \
        --path workspace/$VERSION/$size \
        $FLAGS

    python \
        scripts/get_objectives_of_evaluation_instances.py \
        --path workspace/$VERSION/$size \
        --instances $INSTANCES \
        $FLAGS

    for method in double_sampling_network single_sampling_network column_pre_population nearest_neighbour random_forest lpr coldstart cplex
    do

        python scripts/run_solver.py \
            --path workspace/$VERSION/$size \
            --result-dir evaluation_with_ph \
            --instances $INSTANCES \
            --method $method \
            --ph-type column_evaluation_and_column_combination \
            --tol ${TOL}e-3 \
            --timelimit 10min \
            $FLAGS

        python scripts/run_solver.py \
            --path workspace/$VERSION/$size \
            --result-dir evaluation_without_ph \
            --instances $INSTANCES \
            --method $method \
            --ph-type noop \
            --tol 0e-3 \
            --tol-to-target-objective ${TOL}e-4 \
            --target-objective-file \
                workspace/$VERSION/$size/out/evaluation_target_obj/tol_0.0e+00_primal_noop_dual_cplex_eval.npz \
            --cg-no-progress-action none \
            --timelimit 10min \
            $FLAGS

    done

done
