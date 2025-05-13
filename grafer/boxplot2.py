import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_df = pd.read_csv('../dataset/renad_data.csv')

# Skapa boxplot för endast 'Age'
plt.figure(figsize=(10, 6))
sns.boxplot(x=train_df['Age'])

# Sätt y-axelns maxvärde till 90
plt.ylim(0, 90)

plt.title('Boxplot of Age')
plt.show()

