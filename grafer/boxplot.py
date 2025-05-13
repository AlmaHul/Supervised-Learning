import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

train_df = pd.read_csv('../dataset/renad_data.csv')
sns.boxplot(x='Survived', y='Age', data=train_df)
plt.title("Age vs Survived")
plt.show()
