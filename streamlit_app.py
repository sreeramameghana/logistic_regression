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
      "source": "import streamlit as st\nimport pandas as pd\nimport numpy as np\nfrom sklearn.linear_model import LogisticRegression\nimport joblib\n\n# Load the trained model\nmodel = joblib.load(\"logistic_regression_model.pkl\")  # Ensure this path matches your model file\n\n# Define the Streamlit app\nst.title(\"Titanic Survival Prediction App\")\n\nst.write(\"\"\"\nThis app predicts whether a passenger would have survived the Titanic disaster.\n\"\"\")\n\n# Sidebar for user input features\nst.sidebar.header(\"Input Features\")\n\ndef user_input_features():\n    Pclass = st.sidebar.selectbox(\"Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)\", (1, 2, 3))\n    Sex = st.sidebar.selectbox(\"Sex\", (\"male\", \"female\"))\n    Age = st.sidebar.slider(\"Age\", 0, 100, 29)\n    SibSp = st.sidebar.slider(\"Siblings/Spouses Aboard\", 0, 8, 0)\n    Parch = st.sidebar.slider(\"Parents/Children Aboard\", 0, 6, 0)\n    Fare = st.sidebar.slider(\"Fare\", 0.0, 500.0, 32.0)\n    Embarked = st.sidebar.selectbox(\"Port of Embarkation\", (\"C\", \"Q\", \"S\"))\n\n    data = {\n        \"Pclass\": Pclass,\n        \"Sex\": 0 if Sex == \"male\" else 1,  # Encoding: male=0, female=1\n        \"Age\": Age,\n        \"SibSp\": SibSp,\n        \"Parch\": Parch,\n        \"Fare\": Fare,\n        \"Embarked\": 0 if Embarked == \"C\" else 1 if Embarked == \"Q\" else 2\n    }\n    features = pd.DataFrame(data, index=[0])\n    return features\n\ninput_df = user_input_features()\n\n# Predict using the model\nprediction = model.predict(input_df)\nprediction_proba = model.predict_proba(input_df)\n\n# Display the prediction\nst.subheader(\"Prediction\")\nsurvived = \"Survived\" if prediction[0] == 1 else \"Did Not Survive\"\nst.write(survived)\n\nst.subheader(\"Prediction Probability\")\nst.write(f\"Survived: {prediction_proba[0][1]:.2f}, Did Not Survive: {prediction_proba[0][0]:.2f}\")\n",
      "metadata": {
        "trusted": true
      },
      "outputs": [],
      "execution_count": null
    }
  ]
}