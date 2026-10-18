import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
with open('GroceryStoreDataSet.csv', 'r', encoding='utf-8-sig') as file:
    transactions = [
        [item.strip().strip('"').strip("'")
         for item in line.strip().strip('"').split(',')
         if item.strip().strip('"').strip("'")]
        for line in file
        if line.strip()
    ]
transactions = [t for t in transactions if t]
print("Number of transactions:", len(transactions))
print("First five transactions:")
print(transactions[:5])
te = TransactionEncoder()
encoded_array = te.fit(transactions).transform(transactions)
encoded_df = pd.DataFrame(
    encoded_array,
    columns=te.columns_
)
print("\nEncoded transaction data:")
print(encoded_df.head())
print("\nDataset information:")
encoded_df.info()
print("\nDataset shape:")
print(encoded_df.shape)
support_single_items = encoded_df.sum() / len(encoded_df)
print("\nSupport for single grocery items:")
print(support_single_items.sort_values(ascending=False))
from itertools import combinations
item_pairs = list(combinations(encoded_df.columns, 2))
support_pairs = {}
for pair in item_pairs:
    support_pairs[pair] = (
        encoded_df[list(pair)].all(axis=1).sum()
        / len(encoded_df)
    )
print("\nSupport for pairs of grocery items:")
for pair, support in sorted(
    support_pairs.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{pair}: {support:.3f}")
def generate_association_rules(
    data, min_support=0.05, min_confidence=0.5
):
    frequent_itemsets = apriori(
        data,
        min_support=min_support,
        use_colnames=True
    )
    print("\nFrequent Itemsets:")
    print(frequent_itemsets)

    if frequent_itemsets.empty:
        print(
            "No frequent itemsets found. "
            "Try a lower min_support value, such as 0.01."
        )
        return pd.DataFrame()
    if not frequent_itemsets['itemsets'].apply(
        lambda items: len(items) >= 2
    ).any():
        print("No itemsets containing two or more items were found.")
        return pd.DataFrame()

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )
    return rules
association_rules_df = generate_association_rules(
    encoded_df,
    min_support=0.05,
    min_confidence=0.5
)
print("\nAssociation Rules:")
if association_rules_df.empty:
    print("No association rules found with these settings.")
else:
    print(
        association_rules_df[
            ['antecedents', 'consequents', 'support',
             'confidence', 'lift']
        ].sort_values('confidence', ascending=False)
    )
plt.figure(figsize=(12, 6))
plot = sns.barplot(
    x=support_single_items.index,
    y=support_single_items.values,
    palette='viridis'
)
plt.title('Support of Single Grocery Items')
plt.xlabel('Grocery Items')
plt.ylabel('Support')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Comment about the dataset:
# The Grocery Store dataset contains transactional data of grocery items purchased together. It is useful for market basket analysis to identify relationships between items, which can help in product placement, promotions, and inventory management. The dataset is relatively small and can be easily processed using the Apriori algorithm to find frequent itemsets and generate association rules.