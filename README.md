# **Customer Churn Prediction and Retention Strategy Analysis**

## **Overview**

This initiative focuses on the development of predictive models aimed at identifying telecommunications customers who exhibit a high propensity for churn, defined as the termination of their service agreements. Through rigorous analysis of a comprehensive dataset encompassing customer demographics, subscribed services, and billing information, the primary objective is to derive data-driven insights. These insights are intended to inform the formulation and implementation of effective customer retention strategies. This project meticulously demonstrates a complete data science workflow, spanning from data acquisition and preprocessing to advanced machine learning model development, comprehensive evaluation, and insightful interpretation.

## **Problem Statement**

Customer churn represents a significant operational challenge for telecommunications enterprises, frequently resulting in substantial revenue diminution. The proactive identification of customers likely to churn, prior to their actual service cancellation, enables organizations to deploy targeted retention initiatives. Such initiatives, which may include personalized incentives or enhanced customer support, are instrumental in mitigating churn rates and consequently augmenting customer lifetime value.

## **Dataset**

The data utilized for this project is derived from the **IBM Telco Customer Churn** dataset, which is publicly accessible via Kaggle. This dataset comprises information pertaining to a hypothetical telecommunications company's customer base, encompassing the following categories:

* **Demographic Information:** Gender, SeniorCitizen status, Partner status, and Dependents status.  
* **Service Subscriptions:** Details regarding PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, and StreamingMovies.  
* **Account Information:** Tenure, Contract type, PaperlessBilling status, PaymentMethod, MonthlyCharges, and TotalCharges.  
* **Target Variable:** Churn (binary indicator: Yes/No).

Download Link: https://www.kaggle.com/blastchar/telco-customer-churn  
It is advised to place the downloaded WA\_Fn-UseC\_-Telco-Customer-Churn.csv file within the data/ directory, with an optional renaming to telco\_customer\_churn.csv.

## **Project Structure**

customer-churn-prediction/  
├── data/  
│   └── telco\_customer\_churn.csv  \# Raw dataset  
├── notebooks/  
│   ├── 01\_data\_cleaning\_and\_eda.ipynb            \# Initial data exploration and cleaning  
│   ├── 02\_feature\_engineering.ipynb              \# Preprocessing, encoding, and SMOTE  
│   ├── 03\_model\_training\_and\_evaluation.ipynb    \# Training, hyperparameter tuning, model comparison  
│   └── 04\_insights\_and\_recommendations.ipynb     \# Model interpretability and business recommendations  
├── src/  
│   ├── data\_processor.py         \# Python module for data loading, cleaning, and preprocessing  
│   ├── model\_trainer.py          \# Python module for model training, evaluation, and saving  
│   └── \_\_init\_\_.py               \# Designates 'src' as a Python package  
├── models/  
│   └── best\_churn\_model.pkl      \# Persisted best performing machine learning model  
├── reports/  
│   └── churn\_analysis\_report.pdf \# Placeholder for a summary report or presentation  
├── README.md                     \# Project overview and operational instructions  
├── requirements.txt              \# Enumeration of Python dependencies  
└── LICENSE                       \# Project licensing information

## **Key Capabilities & Technologies**

* **Data Acquisition & Cleaning:** Leveraging Pandas and NumPy for data manipulation.  
* **Exploratory Data Analysis (EDA):** Visualization performed using Matplotlib and Seaborn.  
* **Feature Engineering:** Utilizing Scikit-learn components (OneHotEncoder, StandardScaler) and Imbalanced-learn (SMOTE) for data transformation.  
* **Machine Learning Models:**  
  * Logistic Regression  
  * Random Forest Classifier  
  * XGBoost  
  * LightGBM  
  * Support Vector Classifier (SVC)  
* **Model Evaluation:** Assessment based on Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, and Classification Report.  
* **Hyperparameter Optimization:** Employing GridSearchCV and RandomizedSearchCV for model tuning.  
* **Model Persistence:** joblib for serialization and deserialization of trained models.  
* **Development Environment:** Jupyter Lab/Notebook serves as the primary interactive computing environment.  
* **Version Control:** Git/GitHub for collaborative development and project management.

## **Analysis & Modeling Phases**

### **Data Cleaning & Exploratory Data Analysis (EDA)**

This phase encompasses the loading and initial inspection of the dataset, including examination of its structure and summary statistics. Key activities include the handling of missing values, particularly within the TotalCharges attribute, and the conversion of data types to their appropriate formats. Furthermore, a thorough analysis of the target variable's (Churn) distribution is conducted. Relationships between various features, such as demographic attributes, service subscriptions, and billing information, and their correlation with churn rate are visualized through the application of count plots, histograms, box plots, and correlation matrices.

### **Feature Engineering**

This phase involves the application of several data transformation techniques:

* **Categorical Encoding:** Nominal categorical features are transformed using One-Hot Encoding.  
* **Numerical Scaling:** Numerical features are standardized via StandardScaler to ensure consistent scaling across variables.  
* **Class Imbalance Handling:** To mitigate bias towards the majority class and enhance model performance, SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training data.  
* Subsequent to these transformations, the dataset is partitioned into distinct training and testing subsets.

### **Model Training & Evaluation**

This phase details the systematic approach to model development:

* A diverse suite of classification models, including Logistic Regression, Random Forest, XGBoost, LightGBM, and SVC, is defined for comparative analysis.  
* Hyperparameter optimization is conducted utilizing RandomizedSearchCV (or GridSearchCV for exhaustive exploration) in conjunction with cross-validation to ascertain the optimal parameter configurations for each model.  
* Each model undergoes training on the preprocessed and SMOTE-resampled training data.  
* Model performance is rigorously evaluated on the unseen test set using a comprehensive array of metrics, including Accuracy, Precision, Recall, F1-Score, and ROC-AUC.  
* Visual representations of model performance are generated through Confusion Matrices and ROC Curves, facilitating detailed analytical comparison.  
* The superior performing model is identified based on the ROC-AUC score, given its robustness in handling imbalanced datasets.  
* The selected best model is then persisted for future operational deployment.

### **Insights & Recommendations**

This phase focuses on interpreting the derived analytical findings and translating them into actionable business intelligence:

* **Model Interpretability:** An analysis of feature importances (for tree-based models) or coefficients (for linear models) is conducted to elucidate the factors most influential in predicting customer churn.  
* **Business Insights:** Observations from the EDA and feature importance analysis are synthesized into clear, actionable business insights regarding customer churn drivers.  
* **Retention Strategies:** Concrete strategies are proposed for the telecommunications company, aiming to mitigate churn through targeted promotions, service enhancements, or refined customer engagement programs.

## **Principal Findings & Recommendations**

### **Key Insights:**

* **Contract Type is Critical:** Month-to-month contracts are identified as a primary determinant of customer churn.  
* **Absence of Security and Support Services:** Customers lacking Online Security, Tech Support, and Device Protection services demonstrate an increased susceptibility to churn.  
* **Elevated Monthly Charges:** A direct correlation exists between escalating monthly charges and a higher likelihood of churn, particularly in the absence of long-term contractual commitments.  
* **Significance of Tenure:** Newer customers, characterized by lower tenure, exhibit a heightened risk of churn; conversely, customer loyalty tends to strengthen with increasing tenure.  
* **Fiber Optic Service Discrepancy:** Despite offering high-speed connectivity, subscribers to fiber optic internet service indicate a higher churn rate, potentially attributable to service quality issues or elevated customer expectations.

### **Actionable Recommendations for Customer Retention:**

1. **Promotion of Long-Term Contracts:** Implement incentive programs designed to encourage customers on month-to-month agreements to transition to one-year or two-year contractual plans.  
2. **Bundling of Security & Support Services:** Offer integrated packages or trial periods for security, backup, and technical support services.  
3. **Tiered Pricing & Value Communication:** Clearly articulate the value proposition associated with various pricing tiers, potentially offering loyalty discounts or value-added services to justify higher monthly charges.  
4. **Proactive Early Engagement & Onboarding:** Establish and execute proactive outreach initiatives for new customers, particularly within their initial service period, to ensure satisfaction and address nascent concerns.  
5. **Resolution of Fiber Optic Service Concerns:** Conduct thorough investigations into potential service quality issues associated with fiber optic offerings and implement necessary enhancements.  
6. **Targeted Campaigns for Senior Demographics:** Develop specialized service plans and support options tailored to the unique requirements of senior citizens.  
7. **Personalized Retention Offers:** Leverage the predictive model to identify high-risk customers, facilitating the provision of customized incentives to preempt churn.

## **Operational Procedure**

1. **Repository Cloning:**  
   git clone https://github.com/nicolas0120be/churn-prediction.git
   

2. Dataset Acquisition:  
   Obtain WA\_Fn-UseC\_-Telco-Customer-Churn.csv from Kaggle and deposit it into the data/ directory. An optional renaming to telco\_customer\_churn.csv is permissible.  
3. **Virtual Environment Creation (Recommended):**  
   python \-m venv venv  
   \# For Windows operating systems:  
   .\\venv\\Scripts\\activate  
   \# For macOS/Linux operating systems:  
   source venv/bin/activate

4. **Dependency Installation:**  
   pip install \-r requirements.txt

5. **Jupyter Lab/Notebook Initialization:**  
   jupyter lab

6. Notebook Execution:  
   Navigate to the notebooks/ directory within the Jupyter interface and execute the notebooks in the prescribed sequential order:  
   * 01\_data\_cleaning\_and\_eda.ipynb  
   * 02\_feature\_engineering.ipynb  
   * 03\_model\_training\_and\_evaluation.ipynb  
   * 04\_insights\_and\_recommendations.ipynb

## **Prospective Enhancements**

* **Time-Series Analysis:** Integration of customer behavioral patterns over time to enable dynamic churn predictions.  
* **Advanced Ensemble Methodologies:** Exploration and implementation of stacking or blending techniques involving multiple models.  
* **A/B Testing Simulation:** Modeling the potential impact of A/B testing various customer retention strategies.  
* **Model Deployment:** Operationalization of the optimal model as a web service (e.g., utilizing Flask or FastAPI).  
* **Interactive Dashboard Development:** Creation of a dynamic dashboard (e.g., using Streamlit, Dash) for visualization of churn predictions and analytical insights.
