# Counting French words
import re
from collections import Counter

# Common French words
french_keywords = [" le ", " la ", " et ", " de ", " Bonjour", "bonjour", "merci", "oui", "non", "à ", "être", "être ", "que "]

sample_size = min(100000, len(texts))
sample_texts = texts[:sample_size]

count = 0
hits = Counter()
for t in sample_texts:
    lt = " " + t + " "
    for w in french_keywords:
        if w in lt:
            hits[w.strip()] += 1
            count += 1

print("Sample size:", sample_size)
print("Number of French keyword occurrences (by keyword):", hits)
print("Total number of detected texts (including duplicates):", count)
