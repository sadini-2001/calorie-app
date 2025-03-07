import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import joblib

df = pd.read_csv("data//calories.csv")
df = df.drop(columns=['User_ID'])

X = df.drop(columns=['Calories'])
y = df['Calories']

# using the train test split function
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=104,
                                                    test_size=0.2,
                                                    shuffle=True)

# Convert categorical variables to category dtype
categorical_cols = ['Gender']
numeric_cols = ['Duration', 'Heart_Rate', 'Body_Temp', 'Age', 'Weight', 'Height']

preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', StandardScaler(), numeric_cols),
        ('categorical', OneHotEncoder(), categorical_cols),
    ]
)

# Define pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('svr', SVR(kernel='rbf'))
])

param_grid = {
    'svr__C': [0.01, 0.1, 1, 10, 100],
    'svr__epsilon': [0.01, 0.1, 0.5, 1, 2],
    'svr__gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best cross-validation test MSE:", abs(grid_search.best_score_))

best_model = grid_search.best_estimator_
y_train_pred = best_model.predict(X_train)

train_mse = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
print("Train MSE:", train_mse)
print("Train R^2:", r2_train)

y_pred = best_model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

print('Test MSE:', mse_test)
print('Test R^2:', r2_test)

print('Test MSE:', mse_test)
print('Test R^2:', r2_test)


#save trained model
joblib.dump(best_model,"model/model.pkl")