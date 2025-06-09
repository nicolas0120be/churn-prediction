# src/model_trainer.py

import pandas as pd
import numpy as np
import joblib # For saving/loading models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, classification_report, PrecisionRecallDisplay
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns


class ModelTrainer:
    """
    A class for training, evaluating, and comparing various classification models
    for customer churn prediction.
    """
    def __init__(self, random_state=42):
        """
        Initializes the ModelTrainer.

        Args:
            random_state (int): Seed for reproducibility.
        """
        self.random_state = random_state
        self.models = {}
        self.best_model_name = None
        self.best_model = None
        self.metrics = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])

    def define_models(self):
        """
        Defines a set of classification models to be trained and their parameter grids
        for hyperparameter tuning.
        """
        print("Defining models and their parameter grids...")

        # Logistic Regression
        log_reg_params = {
            'penalty': ['l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear'] # 'liblinear' supports both l1 and l2
        }
        self.models['Logistic Regression'] = {'model': LogisticRegression(random_state=self.random_state, solver='liblinear', max_iter=1000),
                                            'params': log_reg_params}

        # Random Forest
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        self.models['Random Forest'] = {'model': RandomForestClassifier(random_state=self.random_state),
                                        'params': rf_params}

        # XGBoost
        xgb_params = {
            'objective': ['binary:logistic'],
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.2]
        }
        self.models['XGBoost'] = {'model': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss', use_label_encoder=False),
                                  'params': xgb_params}

        # LightGBM
        lgb_params = {
            'objective': ['binary'],
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [-1, 10, 20], # -1 means no limit
            'num_leaves': [31, 50, 70],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'subsample': [0.7, 0.8, 0.9]
        }
        self.models['LightGBM'] = {'model': lgb.LGBMClassifier(random_state=self.random_state),
                                   'params': lgb_params}

        # Support Vector Machine (SVC)
        # Note: SVC can be computationally expensive on large datasets.
        # For simplicity, using a small grid or commenting out if performance is an issue.
        svc_params = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
        self.models['SVC'] = {'model': SVC(random_state=self.random_state, probability=True), # probability=True for ROC-AUC
                              'params': svc_params}

        print(f"Defined {len(self.models)} models.")

    def train_model(self, model_name, X_train, y_train, param_search_type='grid', n_iter_random=50, cv=5):
        """
        Trains a specified model, optionally performing hyperparameter tuning.

        Args:
            model_name (str): The name of the model to train (must be defined in self.models).
            X_train (np.ndarray or pd.DataFrame): Training features.
            y_train (np.ndarray or pd.Series): Training target.
            param_search_type (str): 'grid' for GridSearchCV, 'random' for RandomizedSearchCV.
            n_iter_random (int): Number of parameter settings that are sampled when param_search_type is 'random'.
            cv (int): Number of folds for cross-validation during hyperparameter tuning.

        Returns:
            sklearn.base.BaseEstimator: The best trained model.
        """
        if model_name not in self.models:
            print(f"Error: Model '{model_name}' not defined. Please define it using define_models().")
            return None

        model_info = self.models[model_name]
        estimator = model_info['model']
        param_grid = model_info['params']

        print(f"\n--- Training {model_name} ---")

        if param_grid:
            print(f"Performing hyperparameter tuning using {param_search_type} search...")
            if param_search_type == 'grid':
                search = GridSearchCV(estimator, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
            elif param_search_type == 'random':
                search = RandomizedSearchCV(estimator, param_grid, n_iter=n_iter_random, cv=cv,
                                            scoring='roc_auc', n_jobs=-1, verbose=1, random_state=self.random_state)
            else:
                print("Invalid param_search_type. Using base estimator without tuning.")
                search = estimator

            if param_search_type in ['grid', 'random']:
                search.fit(X_train, y_train)
                best_estimator = search.best_estimator_
                print(f"Best parameters for {model_name}: {search.best_params_}")
                print(f"Best ROC-AUC score on validation sets for {model_name}: {search.best_score_:.4f}")
            else:
                estimator.fit(X_train, y_train)
                best_estimator = estimator
        else:
            print("No parameter grid provided. Training model without tuning.")
            estimator.fit(X_train, y_train)
            best_estimator = estimator

        print(f"{model_name} training complete.")
        return best_estimator

    def evaluate_model(self, model, X_test, y_test, model_name="Current Model"):
        """
        Evaluates a trained model on the test set and prints key metrics.

        Args:
            model (sklearn.base.BaseEstimator): The trained model.
            X_test (np.ndarray or pd.DataFrame): Test features.
            y_test (np.ndarray or pd.Series): Test target.
            model_name (str): Name of the model for reporting.

        Returns:
            dict: A dictionary of evaluation metrics.
        """
        if model is None:
            print(f"Error: Model for {model_name} is None. Cannot evaluate.")
            return {}

        print(f"\n--- Evaluating {model_name} ---")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else [0] * len(y_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob) if hasattr(model, 'predict_proba') else np.nan

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Store metrics
        new_row = pd.DataFrame([{
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        }])
        self.metrics = pd.concat([self.metrics, new_row], ignore_index=True)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

    def plot_confusion_matrix(self, model, X_test, y_test, model_name="Model"):
        """
        Plots the confusion matrix for a trained model.

        Args:
            model (sklearn.base.BaseEstimator): The trained model.
            X_test (np.ndarray or pd.DataFrame): Test features.
            y_test (np.ndarray or pd.Series): Test target.
            model_name (str): Name of the model for the plot title.
        """
        if model is None:
            print(f"Error: Model for {model_name} is None. Cannot plot confusion matrix.")
            return

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {model_name}')
        plt.show()

    def plot_roc_curve(self, models_to_plot, X_test, y_test):
        """
        Plots ROC curves for multiple models on the same graph for comparison.

        Args:
            models_to_plot (dict): A dictionary where keys are model names and values are trained models.
            X_test (np.ndarray or pd.DataFrame): Test features.
            y_test (np.ndarray or pd.Series): Test target.
        """
        plt.figure(figsize=(8, 7))
        plt.plot([0, 1], [0, 1], 'k--', label='No Skill') # Baseline for ROC curve

        for name, model in models_to_plot.items():
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                auc = roc_auc_score(y_test, y_prob)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')
            else:
                print(f"Warning: {name} does not have predict_proba and cannot plot ROC curve.")

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_precision_recall_curve(self, models_to_plot, X_test, y_test):
        """
        Plots Precision-Recall curves for multiple models on the same graph for comparison.

        Args:
            models_to_plot (dict): A dictionary where keys are model names and values are trained models.
            X_test (np.ndarray or pd.DataFrame): Test features.
            y_test (np.ndarray or pd.Series): Test target.
        """
        plt.figure(figsize=(8, 7))

        for name, model in models_to_plot.items():
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
                display = PrecisionRecallDisplay.from_estimator(
                    model, X_test, y_test, name=name, ax=plt.gca()
                )
            else:
                print(f"Warning: {name} does not have predict_proba and cannot plot Precision-Recall curve.")

        plt.title('Precision-Recall Curve Comparison')
        plt.grid(True)
        plt.legend()
        plt.show()


    def get_feature_importance(self, model, feature_names):
        """
        Extracts and displays feature importance for tree-based models or coefficients for linear models.

        Args:
            model (sklearn.base.BaseEstimator): The trained model.
            feature_names (list): List of feature names corresponding to the model's input.

        Returns:
            pd.DataFrame: A DataFrame of feature importances/coefficients.
        """
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
            df_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            df_importances = df_importances.sort_values(by='Importance', ascending=False)
            print("\nFeature Importances (Top 10):")
            print(df_importances.head(10))
            return df_importances

        elif hasattr(model, 'coef_'):
            # Linear models (Logistic Regression, SVC with linear kernel)
            coefs = model.coef_[0] # For binary classification
            df_coefs = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
            df_coefs['Abs_Coefficient'] = np.abs(coefs)
            df_coefs = df_coefs.sort_values(by='Abs_Coefficient', ascending=False)
            print("\nFeature Coefficients (Top 10 by absolute value):")
            print(df_coefs.head(10))
            return df_coefs

        else:
            print("Model does not have feature_importances_ or coef_ attribute for interpretability.")
            return pd.DataFrame()

    def save_model(self, model, path):
        """
        Saves the trained model to a file using joblib.

        Args:
            model (sklearn.base.BaseEstimator): The model to save.
            path (str): The file path to save the model (e.g., 'models/best_churn_model.pkl').
        """
        try:
            joblib.dump(model, path)
            print(f"Model saved successfully to {path}")
        except Exception as e:
            print(f"Error saving model to {path}: {e}")

    def load_model(self, path):
        """
        Loads a trained model from a file using joblib.

        Args:
            path (str): The file path to load the model from.

        Returns:
            sklearn.base.BaseEstimator: The loaded model.
        """
        try:
            model = joblib.load(path)
            print(f"Model loaded successfully from {path}")
            return model
        except FileNotFoundError:
            print(f"Error: Model file not found at {path}.")
            return None
        except Exception as e:
            print(f"An error occurred while loading model from {path}: {e}")
            return None

    def get_best_model_summary(self):
        """
        Returns a summary of the best performing model.
        """
        if self.metrics.empty:
            print("No models have been evaluated yet.")
            return None

        # Sort by ROC-AUC (common for imbalanced classification) or F1-Score
        best_row = self.metrics.sort_values(by='ROC-AUC', ascending=False).iloc[0]
        self.best_model_name = best_row['Model']
        print(f"\nBest Model: {self.best_model_name}")
        print(best_row)
        return best_row

# Example usage (for testing the script directly)
if __name__ == '__main__':
    from data_processor import DataProcessor

    # 1. Load and preprocess data
    data_processor = DataProcessor(data_path='../data/telco_customer_churn.csv')
    data_processor.load_data()
    data_processor.clean_data()
    data_processor.preprocess_data()
    X_train_processed, X_test_processed, y_train, y_test = data_processor.split_data(apply_smote=True)

    if X_train_processed is not None:
        # 2. Initialize ModelTrainer
        trainer = ModelTrainer(random_state=42)
        trainer.define_models()

        trained_models = {}

        # 3. Train each model (you can choose 'grid' or 'random' search)
        for name, model_info in trainer.models.items():
            print(f"\nTraining {name}...")
            # For quick testing, you might reduce n_iter_random or disable tuning
            trained_model = trainer.train_model(name, X_train_processed, y_train,
                                                param_search_type='random', n_iter_random=10)
            if trained_model:
                trained_models[name] = trained_model
                trainer.evaluate_model(trained_model, X_test_processed, y_test, model_name=name)
                trainer.plot_confusion_matrix(trained_model, X_test_processed, y_test, model_name=name)

        # 4. Compare ROC Curves
        if trained_models:
            trainer.plot_roc_curve(trained_models, X_test_processed, y_test)
            trainer.plot_precision_recall_curve(trained_models, X_test_processed, y_test)

        # 5. Get best model and save it
        trainer.get_best_model_summary()
        # Assume 'XGBoost' was the best, this needs to be dynamically determined
        # based on the best_model_name from get_best_model_summary
        # For this example, let's just save the best one if it exists.
        if trainer.best_model_name and trainer.best_model_name in trained_models:
             best_trained_model = trained_models[trainer.best_model_name]
             trainer.save_model(best_trained_model, f'../models/{trainer.best_model_name.replace(" ", "_").lower()}_churn_model.pkl')

             # Optional: Load the model back to test
             loaded_model = trainer.load_model(f'../models/{trainer.best_model_name.replace(" ", "_").lower()}_churn_model.pkl')
             if loaded_model:
                 print("\nVerification: Evaluating loaded model:")
                 trainer.evaluate_model(loaded_model, X_test_processed, y_test, model_name=f"Loaded {trainer.best_model_name}")

        # 6. Feature Importance of the best model (example with a tree-based model)
        if trainer.best_model_name in trained_models and hasattr(trained_models[trainer.best_model_name], 'feature_importances_'):
            # Get feature names from the preprocessor, needed for interpretability
            preprocessor = data_processor.get_preprocessor()
            if preprocessor:
                # The names out from the ColumnTransformer will be in the format 'transformer__feature_name'
                # We need to extract the original feature names.
                # A more robust way:
                # Get the OneHotEncoder's feature names
                ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(
                                        data_processor.categorical_features
                                    ).tolist()
                # Combine numerical and one-hot encoded names
                processed_feature_names = data_processor.numerical_features + ohe_feature_names
                trainer.get_feature_importance(trained_models[trainer.best_model_name], processed_feature_names)

