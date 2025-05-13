from sklearn.model_selection import train_test_split
import pandas as pd

train_df = pd.read_csv('../dataset/renad_data.csv')

# Välj features (X) och målvariabel (y)
X = train_df.drop(['Survived', 'Name', 'Ticket', 'PassengerId'], axis=1)
y = train_df['Survived']

# Dela upp i träningsdata (80%) och testdata (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)
y_pred_forest = forest_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_forest))
