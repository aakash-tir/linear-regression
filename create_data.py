import numpy as np
import pandas as pd

#set seed
np.random.seed(369)

# Generate 100 x values between 0 and 20
x = np.linspace(0, 20, 100)

#define function
noise = np.random.normal(0, 10, size=x.shape)  # mean=0, std=2
y = np.round(np.random.uniform(5, 10), 4) * x + np.round(np.random.uniform(0, 5), 4) + noise

#dataframe creation
df = pd.DataFrame({'x': x, 'y': y})

#save as csv
df.to_csv('linear_data.csv', index=False)

print("Dataset saved to 'linear_data.csv'")
