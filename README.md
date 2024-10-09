## Diabetes Prediction Machine Learning Model

This project aims to develop a machine learning model capable of accurately predicting the likelihood of diabetes in individuals based on various medical and lifestyle factors. By early detection, this model can assist healthcare professionals in providing timely interventions and improving patient outcomes.

## Data

The dataset used in this project was sourced from Kaggle and contains medical records for individuals. The key features used for prediction include:

- Gender
- Age
- Hypertension (High Blood Pressure)
- Heart Disease
- Smoking History
- BMI
- HbA1c Level (Blood Sugar)
- Blood Glucose Level

The target variable is "diabetes," indicating whether an individual has diabetes or not.

## Model

This project explored various machine learning models for predicting diabetes. The models evaluated include:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

The models were evaluated using the F1 score, a metric that considers both precision and recall. The XGBoost Classifier achieved the highest F1 score and was selected as the best performing model.

**Model Evaluation on Training Data**

- F1 Score: 0.990021
- Accuracy: 0.989921
- AUC: 0.999539

**Model Evaluation on Test Data**

The XGBoost classifier was evaluated on an unseen test dataset to assess its generalizability. The following metrics were obtained:

- Precision: 0.815642
- Recall: 0.768421
- F1 Score: 0.791328
- Accuracy: 0.964808
- AUC: 0.875952

**Confusion Matrix**

| Predicted | Negative             | Positive            |
| --------- | -------------------- | ------------------- |
| Actual    | True Negative (1965) | False Positive (33) |
| Actual    | False Negative (44)  | True Positive (146) |

## Feature Importance

The XGBoost model identified the following features as the most significant predictors of diabetes status:

- **BMI**
- **Age**
- **Blood Glucose Level**
- **HbA1c Level**

These findings align with existing medical knowledge about risk factors for diabetes.

![feature importance](https://github.com/duyilemi/diabetes_prediction_model/blob/master/feature_importance.png?raw=true)

This visualization further emphasizes the importance of these features in predicting diabetes.

## Streamlit App

This project also includes a Streamlit application that allows users to interact with the predictive model. Users can input their medical information, and the app will predict the likelihood of diabetes. Additionally, the app can log user input data and prediction results to a Google Sheet for further analysis.

You can access the Streamlit app here: [https://diabetesmlpredictionmodel.streamlit.app](https://diabetesmlpredictionmodel.streamlit.app)

## Future Work

Future work could involve:

- Collecting a larger and more diverse dataset to improve model generalizability.
- Fine-tuning the hyperparameters of the XGBoost classifier for potentially better performance.
- Integrating the model into a clinical decision support system.

This project demonstrates the potential of machine learning for predicting diabetes and supporting early detection efforts. However, it's important to emphasize that this model is intended for informational purposes only and should not be used for medical diagnosis. Always consult a healthcare professional for any medical concerns.
