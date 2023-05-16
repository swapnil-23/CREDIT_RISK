import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder

# Load the training data
df = pd.read_csv('credit_customers.csv')

# Drop the 'num_dependents' column
df.drop(['num_dependents'], axis=1, inplace=True)

# Create an instance of OrdinalEncoder
ordinal_encoder = OrdinalEncoder()

# Fit and transform the dataframe using the encoder
df_encoded = ordinal_encoder.fit_transform(df)

# Save the OrdinalEncoder object to a file
with open('ordinal_encoder.pkl', 'wb') as file:
    pickle.dump(ordinal_encoder, file)
