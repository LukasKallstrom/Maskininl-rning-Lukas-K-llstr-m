import joblib
import pandas as pd

model = joblib.load('cardio_model.pkl')

df = pd.read_csv('test_samples.csv')

prediction = model.predict(df.drop('cardio', axis=1))

pred_df = pd.DataFrame()
pred_df["probability_class_0"] = model.predict_proba(df.drop('cardio', axis=1))[:,0]
pred_df["probability_class_1"] = model.predict_proba(df.drop('cardio', axis=1))[:,1]
pred_df["prediction"] = prediction

pred_df.to_csv('predictions.csv', index=False)
