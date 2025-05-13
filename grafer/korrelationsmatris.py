import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

train_df = pd.read_csv('../dataset/renad_data.csv')

# Filtrera fram endast numeriska kolumner
numeric_df = train_df.select_dtypes(include=['number', 'bool'])

# Skapa korrelationsmatris
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
