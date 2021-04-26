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

allMovies = pd.read_csv("./dataset/movies.csv")
allRatings = pd.read_csv("./dataset/ratings.csv")
positiveRatings = allRatings[allRatings.rating >= 3]
positiveRatings = positiveRatings[positiveRatings.groupby('movieId')['rating'].transform('count').ge(30)]


print(positiveRatings)