import pandas as pd
df=pd.read_csv('modifiedTiatanic3.csv')
df = df.drop(['Name','Ticket'], axis=1)
# Convert Categorical Variables to Numeric: Use one-hot encoding or label encoding for categorical variables:
df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Pclass', 'AgeGroup', 'FareGroup'], drop_first=True)


#Separate the data into training and testing sets:
from sklearn.model_selection import train_test_split

X = df.drop('Survived', axis=1)  # Features
y = df['Survived']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#. Model Building
#Train machine learning models to predict survival:

# Logistic Regression:

# from sklearn.linear_model import LogisticRegression

# model = LogisticRegression()
# model.fit(X_train, y_train)

# Random Forest:
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# 4. Evaluate the Model
# Assess model performance using metrics like accuracy, precision, recall, and F1 score:
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 5. Fine-Tune the Model
# Use hyperparameter tuning to improve performance:

# GridSearchCV for hyperparameter optimization:

from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

#  . Evaluate Model on Test Data
# Reassess the final model's performance using the test data to ensure generalization:
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_test_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))


# Analyze feature importance
import matplotlib.pyplot as plt
import pandas as pd

# Example for Random Forest
feature_importance = model.feature_importances_
features = X_train.columns

# Plot feature importance
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# 3. Save the Model
import joblib

# Save model
joblib.dump(model, 'titanic_model.pkl')

# Load model
loaded_model = joblib.load('titanic_model.pkl')
