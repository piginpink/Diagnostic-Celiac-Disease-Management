# Importing all the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# reading the dataset
file_path = r"C:\Users\jessi\OneDrive - Princeton University\Princeton Hacks\Diagnostic-Celiac-Disease-Management\PCOS Dataset.csv"
data = pd.read_csv(file_path)

# Cleaning data set (converting strings to numbers)
data["AMH(ng/mL)"] = pd.to_numeric(data["AMH(ng/mL)"], errors='coerce')
data["II    beta-HCG(mIU/mL)"] = pd.to_numeric(data["II    beta-HCG(mIU/mL)"], errors='coerce')

data['Marraige Status (Yrs)'] = data['Marraige Status (Yrs)'].fillna(data['Marraige Status (Yrs)'].median())
data['II    beta-HCG(mIU/mL)'] = data['II    beta-HCG(mIU/mL)'].fillna(data['II    beta-HCG(mIU/mL)'].median())
data['AMH(ng/mL)'] = data['AMH(ng/mL)'].fillna(data['AMH(ng/mL)'].median())
data['Fast food (Y/N)'] = data['Fast food (Y/N)'].fillna(data['Fast food (Y/N)'].mode()[0])

# Clearing up the extra space in the column names (optional)
data.columns = [col.strip() for col in data.columns]

# Identifying non-numeric columns
non_numeric_columns = data.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_columns)

# Converting non-numeric columns to numeric where possible
for col in non_numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Dropping rows with any remaining non-numeric values
data.dropna(inplace=True)

# Preparing data for model training
X = data.drop(["PCOS (Y/N)", "Sl. No", "Patient File No.", "Marraige Status (Yrs)", "Blood Group","II    beta-HCG(mIU/mL)","TSH (mIU/L)","Waist:Hip Ratio"], axis=1) # dropping out index from features too
y = data["PCOS (Y/N)"]

# Splitting the data into test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Fitting the RandomForestClassifier to the training set
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Making prediction and checking the test set
pred_rfc = rfc.predict(X_test)
accuracy = accuracy_score(y_test, pred_rfc)
print(accuracy)

# Example: Collecting user input for the features
print("Please enter the following details:")

user_input = np.array([4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 55, 5, 5, 5, 5, 5, 5, 5])

# Reshape the user input to be 2D
user_input_reshaped = user_input.reshape(1, -1)

# Convert user input to DataFrame with feature names
user_input_df = pd.DataFrame(user_input_reshaped, columns=X.columns)

# Assuming the scaler used during training
scaler = StandardScaler()  # Normally you'd load a fitted scaler

# For demonstration, let's assume no scaling (remove if you're scaling the input)
user_input_scaled = user_input_df  # Remove this if you're applying scaling

# Step 3: Get the probability of PCOS
probabilities = rfc.predict_proba(user_input_scaled)

# Extract probability for PCOS (class 1)
probability_pcos = probabilities[0][1]  # Probability of PCOS (class 1)
probability_non_pcos = probabilities[0][0]  # Probability of non-PCOS (class 0)

# Step 4: Output the result
print(f"Probability of PCOS: {probability_pcos * 100:.2f}%")
print(f"Probability of non-PCOS: {probability_non_pcos * 100:.2f}%")