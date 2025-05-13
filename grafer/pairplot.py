import seaborn as sns
import pandas as pd

train_df = pd.read_csv('../dataset/renad_data.csv')


# Konvertera boolska kolumner till numeriska (True blir 1, False blir 0)
train_df['Sex_male'] = train_df['Sex_male'].astype(int)

# Gör pairplot med numeriska kolumner
sns.pairplot(train_df[['Survived', 'Age', 'Fare', 'Pclass', 'Sex_male']], hue='Survived')
