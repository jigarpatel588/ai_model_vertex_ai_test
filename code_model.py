import pandas as pd
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load data from BigQuery
client = bigquery.Client()
query = """
SELECT * FROM `superb-reporter-430115-t3.sale_output.super_store_sales`
"""
df = client.query(query).to_dataframe()

# Feature engineering
X = df.drop('Sales', axis=1)
y = df['Sales']
# display(X)
# display(y)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train an XGBoost model
model = XGBRegressor(n_estimators=100)
display(X_train)
display(y_train)

model.fit(X_train._get_numeric_data(), y_train._get_numeric_data())

# # Evaluate the model
predictions = model.predict(X_test._get_numeric_data())
print(f'Mean Squared Error: {mean_squared_error(y_test, predictions)}')
