import matplotlib.pyplot as plt
import pandas as pd

train_df = pd.read_csv('renad_data.csv')

train_df['Age'].hist(bins=30)
plt.title("Age distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()