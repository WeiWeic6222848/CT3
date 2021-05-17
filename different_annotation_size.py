# this file is added for extra experiment - CT3
from collections import defaultdict

import lib
from lib.recommenders import *
import pandas as pd

BASELINES = [DebiasedModel]
OUTFILE = "results/3_different_size.json"
INFILE = "results/2a_find_best.json"

dataset = lib.ImplicitMovieLens("ml-25m-implicit")
metrics = [lib.eval.SumOfRanks(), lib.eval.RecallAtK(
    100), lib.eval.RecallAtK(50), lib.eval.RecallAtK(25)]

supervised_testset = dataset.load_rankings(
    "datasets/labeled/similarity_judgements.test.csv", "movieId", "neighborId",
    "sim_bin", verbose=True)

results = dict()
table = list()

prev_models = lib.from_json(INFILE)
for size in [20, 50, 70, 100, 200, 380]:
    supervised_trainset = dataset.load_rankings(
        "datasets/labeled/similarity_judgements.train.csv", "movieId",
        "neighborId", "sim_bin", verbose=True, limitAmount=size)

    for model in BASELINES:
        avgMetrics = defaultdict(list)
        examples = None

        model_name = model.__name__
        print("\nCurrent model " + model_name + " with " + str(
            size) + " positive pairs of annotation data")

        if model_name in prev_models:
            best_param = prev_models[model_name]["best_param"]
            # print("Best params on val:", best_param)
        else:
            best_param = dict()
            # print("No hyperparameters found.")

        for time in range(5):

            best_model = model(dataset, supervised_trainset,**best_param)
            best_model.fit()
            ms = best_model.evaluate(supervised_testset, metrics)
            for m in ms.keys():
                avgMetrics[m].append(ms[m]['mean'])

            examples = best_model.get_scored_candidates(
                0).to_dict('records')

        for m in avgMetrics.keys():
            avgMetrics[m] = sum(avgMetrics[m]) / len(avgMetrics[m])
        results[str(size)] = avgMetrics
        results[str(size)]["example"] = examples

        for metric, scores in results[str(size)].items():
            if metric != "example":
                table += [{
                    "model": str(size), "metric": metric,
                    "mean score": scores, "size": size}]

lib.to_json(OUTFILE, results)
print(pd.DataFrame(table).to_string())
