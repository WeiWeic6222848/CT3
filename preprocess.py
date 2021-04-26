import pandas as pd

seedMovies = pd.read_csv("./dataset/test-set.csv")

pairJudgement = pd.read_csv("./dataset/pair-responses.csv")
# print(pairJudgement)
positiveJudgement = pairJudgement.groupby(by=["movieId", "neighborId"]) \
    .filter(lambda x: len(x) >= 2) \
    .groupby(by=["movieId", "neighborId"]) \
    .mean()

positiveJudgement = positiveJudgement[positiveJudgement.sim >= 2].groupby(
    by=["movieId"]).filter(lambda x: len(x) > 4)

print(positiveJudgement.groupby(by=["movieId"]).count().mean())
