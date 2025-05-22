import numpy as np
import pandas as pd

# Set seed
np.random.seed(42)


x = np.random.uniform(0, 10, 100)
y = np.random.uniform(0, 10, 100)

# True parameters
a = 3.5   
b = -2.0  
c = 5.0   

# add noise
noise = np.random.normal(0, 2, size=100)
z = a * x + b * y + c + noise

# save a dataframe in csv
df = pd.DataFrame({'x': x, 'y': y, 'z': z})
df.to_csv('data/3d_linear_data.csv', index=False)

print("3D linear dataset saved to '3d_linear_data.csv'")
