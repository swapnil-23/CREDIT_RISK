import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
import pickle


df = pd.read_csv('D:\CREDIT_RISK\credit_customers.csv')

df.drop(['num_dependents'], axis=1, inplace=True)

ord_enc = OrdinalEncoder()

df['checking_status'] = ord_enc.fit_transform(df[['checking_status']])
df['credit_history'] = ord_enc.fit_transform(df[['credit_history']])
df['purpose'] = ord_enc.fit_transform(df[['purpose']])
df['savings_status'] = ord_enc.fit_transform(df[['savings_status']])
df['employment'] = ord_enc.fit_transform(df[['employment']])
df['personal_status'] = ord_enc.fit_transform(df[['personal_status']])
df['other_parties'] = ord_enc.fit_transform(df[['other_parties']])
df['property_magnitude'] = ord_enc.fit_transform(df[['property_magnitude']])
df['other_payment_plans'] = ord_enc.fit_transform(df[['other_payment_plans']])
df['housing'] = ord_enc.fit_transform(df[['housing']])
df['job'] = ord_enc.fit_transform(df[['job']])
df['own_telephone'] = ord_enc.fit_transform(df[['own_telephone']])
df['foreign_worker'] = ord_enc.fit_transform(df[['foreign_worker']])
df['class'] = ord_enc.fit_transform(df[['class']])

## splitting the dataset

X_train, X_test, y_train, y_test = train_test_split(df.drop('class', axis=1), df['class'], test_size=0.2, random_state=42)


## ML model

model = XGBClassifier(
    learning_rate=0.20,
    n_estimators=300,
    max_depth=4,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=60,
    eval_metric='error',
)


pickle.dump(XGBClassifier, open("model.pkl", "wb"))