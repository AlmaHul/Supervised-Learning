import pandas as pd
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('../dataset/renad_data.csv')

# Välj features (X) och målvariabel (y)
X = train_df.drop(['Survived', 'Name', 'Ticket', 'PassengerId'], axis=1)
y = train_df['Survived']


from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Skalning av data (PCA fungerar bäst på skalerade data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applicera PCA (t.ex. reducera till 2 huvudkomponenter)
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)

# Dela upp i tränings- och testdata
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Träna Logistisk Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Prediktera och utvärdera
y_pred_log = log_model.predict(X_test)
print("Logistic Regression with PCA Accuracy:", accuracy_score(y_test, y_pred_log))

pca = PCA()
pca.fit(X_scaled)



# Hur många komponenter bör användas?
import matplotlib.pyplot as plt
plt.plot(range(1, len(pca.explained_variance_ratio_)+1),
         pca.explained_variance_ratio_.cumsum(), marker='o')
plt.xlabel('Antal komponenter')
plt.ylabel('Kumulativ förklarad varians')
plt.title('PCA - Varians förklarad per komponent')
plt.grid(True)
plt.show()
