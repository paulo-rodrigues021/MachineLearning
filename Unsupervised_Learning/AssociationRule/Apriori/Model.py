
# importing the data (csv format)
import pandas as pd
base = pd.read_csv('groceries.csv', sep=',', header=None)

# transforming each database's line into an array
transactions = []
for i in range(len(base)):
    transactions.append([str(base.values[i, j]) for j in range(len(base.columns))])

# Importing the rule algorithm and setting the parameters
from apyori import apriori
rules = apriori(transactions, min_support=0.003, min_confidence=0.6, min_lift=2.0, min_lenght=2)
results = list(rules)

# Listing the main 5 rules found
results = [list(x) for x in results]
shapedResults = []
for i in range(5):
    shapedResults.append([list(x) for x in results[i][2]])
print(shapedResults)

