import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("coal_hc_data.csv")
features = ['ash_content', 'volatile_matter', 'carbonization_temp', 'carbonization_time', 'heating_rate']
X = df[features].fillna(df[features].median())
y = df['reversible_capacity']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
model.fit(X_scaled, y)

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(features, "features.pkl")

print("模型训练完成，已保存")
