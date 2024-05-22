import numpy as np

# Set the random seed once
np.random.seed(42)
choices1 = np.random.choice([1, 2, 3, 4, 5], size=3)
print("First set of choices:", choices1)

np.random.seed(42)
choices2 = np.random.choice([1, 2, 3, 4, 5], size=3)
print("Second set of choices:", choices2)
