# this file is added for clean plots and tables - CT3
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt

resultB = pd.read_json("results/2b_eval_on_test.json")
resultC = pd.read_json("results/3_different_size.json")

scores = resultB.keys()

for algorithm in resultB.columns:
    for key in resultB[algorithm].keys():
        if key != 'example':
            resultB[algorithm][key] = resultB[algorithm][key]['mean']

Table2 = resultB.drop(["example"])
print(tabulate(Table2, headers='keys', tablefmt='psql'))

Table3 = resultB.loc["example"].to_frame().T
examples = Table3[::1]
Table3 = Table3.drop(["example"])
for i in range(10):
    rank = i + 1
    items = {}
    for key in examples:
        items[key] = examples[key][0][i]['title']
    Table3 = Table3.append(pd.Series(items, name="Rank {}".format(str(rank))))
print(tabulate(Table3, headers='keys', tablefmt='psql'))

recall100 = resultC.loc["RecallAt100"]
recall50 = resultC.loc["RecallAt50"]
recall25 = resultC.loc["RecallAt25"]
y = resultC.columns
l1 = plt.plot(y, recall100, 'bo--', label="recall@100")
l2 = plt.plot(y, recall50, 'gs--', label="recall@50")
l3 = plt.plot(y, recall25, 'x--', label="recall@25", color='orange')
plt.legend()
plt.xlabel("# of annotated movie pairs for training")
plt.ylabel("mean score")
plt.show()

ranking = resultC.loc["SumOfRanks"]
plt.plot(y, ranking, 'r+--', label="recall@100")
plt.xlabel("# of annotated movie pairs for training")
plt.show()
