import pandas as pd

import os

from sklearn.model_selection import train_test_split


def get_reliable_responses(responses, ratings, movies):
    top_2500_ids = ratings.movieId.value_counts().index[:2500].tolist()
    inventory_ids = set(movies.movieId.tolist())
    responses["seedValid"] = responses.movieId.isin(top_2500_ids)
    responses["neighborValid"] = responses.neighborId.isin(inventory_ids)
    responses["valid"] = responses["seedValid"] & responses["neighborValid"]

    reliable_responses = \
        responses[responses.valid].groupby(["movieId", "neighborId"]).filter(
            lambda x: len(x) >= 2)[
            ["movieId", "neighborId", "sim", "goodRec"]]
    return reliable_responses.groupby(
        ["movieId", "neighborId"]).mean().sort_values(by=['movieId', 'sim'],
                                                      ascending=[True, False])


class dataLoader:
    datasetPath = os.path.join(".", "dataset")

    def __init__(self):
        seedMovies = pd.read_csv(
            os.path.join(self.datasetPath, "test-set.csv"))

        pairJudgement = pd.read_csv(
            os.path.join(self.datasetPath, "pair-responses.csv"))

        allMovies = pd.read_csv(
            os.path.join(self.datasetPath, "movies.csv"))
        allRatings = pd.read_csv(
            os.path.join(self.datasetPath, "ratings.csv"))

        pairJudgement = get_reliable_responses(pairJudgement, allRatings,
                                               allMovies)

        # positiveJudgement = pairJudgement.groupby(by=["movieId", "neighborId"]) \
        #     .filter(lambda x: len(x) >= 2) \
        #     .groupby(by=["movieId", "neighborId"]) \
        #     .mean()

        # record positive pairs
        pairJudgement["positive"] = pairJudgement.sim.ge(
            2)

        # filter out seedmovies with less than 5 positive pairs (at least 4)
        pairJudgement = pairJudgement.groupby(
            by=["movieId"]).filter(lambda x: len(x[x["positive"] == True]) > 4)

        print(len(pairJudgement.groupby(by=["movieId"])))
        print(pairJudgement.groupby(by=["movieId"]).count().mean())

        # positiveRatings = allRatings[allRatings.rating >= 3]
        # positiveRatings = positiveRatings[
        #     positiveRatings.groupby('movieId')['rating'].transform('count').ge(
        #         30)]
        # print(positiveRatings)

        # print(positiveJudgement)

        self.XTrain, self.Ytrain, XTest, YTest = train_test_split(
            pairJudgement,
            pairJudgement["positive"],
            test_size=33 / 67,
            random_state=42)
        self.XTest, self.YTest, self.XValidation, self.YValidation = train_test_split(
            XTest,
            YTest,
            test_size=16 / 33,
            random_state=42)

        print(
            "loaded the dataset and splitted into Train, Test and Validation Set")
        print("train set size: ", len(self.XTrain))
        print("test set size: ", len(self.XTest))
        print("validation set size: ", len(self.XValidation))


dataLoader()
