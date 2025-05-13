from sklearn.model_selection import train_test_split
import pandas as pd

train_df = pd.read_csv('../dataset/renad_data.csv')

# Välj features (X) och målvariabel (y)
X = train_df.drop(['Survived', 'Name', 'Ticket', 'PassengerId'], axis=1)
y = train_df['Survived']

# Dela upp i träningsdata (80%) och testdata (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import numpy as np
from scipy import stats

# Lägg till z-score filter för att ta bort outliers i 'Age'
z_scores = np.abs(stats.zscore(X['Age']))
X_no_outliers = X[(z_scores < 3)]
y_no_outliers = y[X_no_outliers.index]

# Dela upp i tränings- och testdata igen efter att ha tagit bort outliers
X_train_no_outliers, X_test_no_outliers, y_train_no_outliers, y_test_no_outliers = train_test_split(X_no_outliers, y_no_outliers, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Träna Random Forest-modellen utan outliers
forest_model_no_outliers = RandomForestClassifier(n_estimators=100, random_state=42)
forest_model_no_outliers.fit(X_train_no_outliers, y_train_no_outliers)

# Gör prediktioner och utvärdera
y_pred_forest_no_outliers = forest_model_no_outliers.predict(X_test_no_outliers)
print("Random Forest Accuracy without Outliers:", accuracy_score(y_test_no_outliers, y_pred_forest_no_outliers))
