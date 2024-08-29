{
  "metadata": {
    "kernelspec": {
      "name": "python",
      "display_name": "Python (Pyodide)",
      "language": "python"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "python",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8"
    }
  },
  "nbformat_minor": 4,
  "nbformat": 4,
  "cells": [
    {
      "cell_type": "code",
      "source": "import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import LabelEncoder\nimport joblib\n\nimport pandas as pd\n\n# Load the dataset\ntitanic_data = pd.read_csv('Titanic_train.csv')  # Replace with your file path\n\nfrom sklearn.preprocessing import LabelEncoder\n\n# Function to clean the data\ndef clean_data(data):\n    # Fill missing age values with the median age\n    data['Age'] = data['Age'].fillna(data['Age'].median())\n\n    # Fill missing embarked values with the most frequent value\n    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])\n\n    # Fill missing fare values with the median fare (if applicable)\n    if 'Fare' in data.columns:\n        data['Fare'] = data['Fare'].fillna(data['Fare'].median())\n\n    # Create a new feature indicating whether a cabin number is known\n    data['CabinKnown'] = data['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)\n    data = data.drop('Cabin', axis=1)\n    \n    return data\n\n# Function to encode categorical variables\ndef engineer_and_encode_features(data):\n    # Extract titles from the name\n    data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())\n    \n    # Create family size feature\n    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1\n\n    # Label encode categorical variables\n    label_encoders = {}\n    categorical_columns = ['Sex', 'Embarked', 'Title']\n    \n    for column in categorical_columns:\n        label_encoders[column] = LabelEncoder()\n        data[column] = label_encoders[column].fit_transform(data[column])\n\n    # Drop columns that are not needed\n    data = data.drop(['Name', 'Ticket'], axis=1)\n    \n    return data\n\n# Apply data cleaning and feature engineering to the dataset\ntitanic_data_cleaned = clean_data(titanic_data)\ntitanic_data_final = engineer_and_encode_features(titanic_data_cleaned)\n\n\n# Assuming you have your cleaned and encoded data in 'titanic_data_final'\n\n# Step 1: Prepare the Data\nX = titanic_data_final.drop('Survived', axis=1)  # Features\ny = titanic_data_final['Survived']               # Target variable\n\n# Step 2: Split the Data (Optional: if you want to hold out a validation set)\nX_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)\n\n# Step 3: Train the Logistic Regression Model\nmodel = LogisticRegression(max_iter=1000, random_state=42)\nmodel.fit(X_train, y_train)\n\n# Step 4: Save the Trained Model to a File\njoblib.dump(model, 'logistic_regression_model.pkl')\n",
      "metadata": {
        "trusted": true
      },
      "outputs": [
        {
          "name": "stderr",
          "text": "<ipython-input-1-3de53fff3080>:1: DeprecationWarning: \nPyarrow will become a required dependency of pandas in the next major release of pandas (pandas 3.0),\n(to allow more performant data types, such as the Arrow string type, and better interoperability with other libraries)\nbut was not found to be installed on your system.\nIf this would cause problems for you,\nplease provide us feedback at https://github.com/pandas-dev/pandas/issues/54466\n        \n  import pandas as pd\n/lib/python3.12/site-packages/sklearn/linear_model/_logistic.py:469: ConvergenceWarning: lbfgs failed to converge (status=1):\nSTOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n\nIncrease the number of iterations (max_iter) or scale the data as shown in:\n    https://scikit-learn.org/stable/modules/preprocessing.html\nPlease also refer to the documentation for alternative solver options:\n    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n  n_iter_i = _check_optimize_result(\n",
          "output_type": "stream"
        },
        {
          "execution_count": 1,
          "output_type": "execute_result",
          "data": {
            "text/plain": "['logistic_regression_model.pkl']"
          },
          "metadata": {}
        }
      ],
      "execution_count": 1
    },
    {
      "cell_type": "code",
      "source": "",
      "metadata": {
        "trusted": true
      },
      "outputs": [],
      "execution_count": null
    }
  ]
}