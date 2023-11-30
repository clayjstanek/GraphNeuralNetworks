########## cstanek@camgian.com  10/24/23

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

p = Path(__file__)
p = p.parents[0]
filepath = p.joinpath("sim_matrix_24hr.pkl")

with open(filepath, 'rb') as file:
    data = pickle.load(file)
    data = np.array(data, dtype = 'float32')
    similarity_data = data

# Flatten the 2D array to 1D
flattened_data = similarity_data.flatten()

# Create histogram
plt.hist(flattened_data, bins=100, color='blue', edgecolor='black')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Values of the 24 Hour Similarity Matrix')

# Display the plot
plt.show()

"""
In this example:

The 2D array is flattened to a 1D array to make it suitable for histogram plotting.
plt.hist() function is used to create the histogram. The bins parameter is 
set to 100 as per requirement.
Labels and title are added to make the plot more informative.
Finally, plt.show() is called to display the plot.
"""





