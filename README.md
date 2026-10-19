 Water Quality Prediction

 Project Overview : 
 
This project applies machine learning techniques to predict whether water is potable (safe for human consumption) based on its chemical properties.  
It demonstrates a complete data science workflow, including data preprocessing, exploratory analysis, model development, and deployment through an interactive dashboard.

 Problem Statement: 
Access to clean drinking water remains a critical global challenge. Traditional laboratory testing methods, while accurate, are often time‑consuming and resource‑intensive.  
Objective: Develop a predictive model that classifies water samples as safe or unsafe using measurable parameters such as pH, hardness, turbidity, and sulfate levels.  
This solution aims to provide a faster, cost‑effective, and scalable approach to water quality assessment.

 Dataset
- Source: [Kaggle Water Quality Dataset](https://www.kaggle.com/datasets/developerghost/water-potability)  
- Features: pH, hardness, TDS, chlorine, sulfate, conductivity, organic carbon, trihalomethanes, turbidity  
- Target: Potability (0 = unsafe, 1 = safe)

 Exploratory Data Analysis (EDA)
- Checked dataset size, column types, and summary statistics.
- Found missing values in pH, TDS, sulfate, and conductivity.
- Visualized distributions, boxplots, and correlations.

Data Cleaning
Filled missing values using mean or median to ensure completeness.
code:
df['ph'] = df['ph'].fillna(df['ph'].mean())
df['tds'] = df['tds'].fillna(df['tds'].median())
df['sulfate'] = df['sulfate'].fillna(df['sulfate'].mean())
df['conductivity'] = df['conductivity'].fillna(df['conductivity'].median())

Model Training
We trained two baseline models: Logistic Regression and Random Forest.
Code:
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

Result :
<img width="524" height="455" alt="Untitled" src="https://github.com/user-attachments/assets/5e683ea8-470e-43ea-bfd6-da359b0518b1" />
<img width="536" height="547" alt="Untitled-1" src="https://github.com/user-attachments/assets/d4fb08f2-0fc0-44f3-ba2a-5f4b2fa4db1d" />

Logistic Regression Accuracy: 0.9252
Random Forest Accuracy: 0.99745
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     18524
           1       0.97      1.00      0.98      1476

    accuracy                           1.00     20000
   macro avg       0.98      1.00      0.99     20000
weighted avg       1.00      1.00      1.00     20000


Code :
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score

 Confusion Matrix :
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot(cmap='Blues')


 ROC Curve :
y_prob_rf = rf_model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob_rf)
auc_score = roc_auc_score(y_test, y_prob_rf)

Insights Section  :
Turbidity, pH, and sulfate are strong predictors of potability.
Random Forest consistently outperformed Logistic Regression.

Next Steps Section  
Hyperparameter tuning (GridSearchCV).
Deploy a Streamlit dashboard for interactive predictions.

 Full Notebook:
For complete code, detailed EDA, and all plots, see the  
[Jupyter Notebook](https://github.com/liyajohn627-rgb/Water-quality-prediction-/commit/48ed3e2a1e38c9a537702904aa28c86651767649#diff-4405df7d92e6e777dd6a948357ac165590d1bfd20559888cff87c4c3a85ee292).

Requirements:

pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib



Author:

Liya John
B.Tech AI & Data Science Student
Interested in machine learning, data analysis, and community projects.
