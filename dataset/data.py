import pandas as pd

# Läs in tränings- och testdata
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


# Kolla hur många saknade värden det finns i varje kolumn
#print(train_df.isnull().sum())


train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)


# Ta bort Cabin-kolumnen
train_df.drop('Cabin', axis=1, inplace=True)


train_df = pd.get_dummies(train_df, columns=['Sex', 'Embarked'], drop_first=True)

train_df.info()

train_df.to_csv("renad_data.csv", index=False)


