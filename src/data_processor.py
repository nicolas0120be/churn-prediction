# src/data_processor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

class DataProcessor:
    """
    A class for loading, cleaning, preprocessing, and splitting
    the Telco Customer Churn dataset.
    """
    def __init__(self, data_path='data/telco_customer_churn.csv'):
        """
        Initializes the DataProcessor with the path to the dataset.

        Args:
            data_path (str): The path to the CSV dataset.
        """
        self.data_path = data_path
        self.df = None
        self.preprocessor = None # To store the ColumnTransformer/Pipeline
        self.X = None
        self.y = None

    def load_data(self):
        """
        Loads the dataset from the specified path.

        Raises:
            FileNotFoundError: If the data file is not found.
            Exception: For other errors during data loading.
        """
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully from {self.data_path}")
            print(f"Initial shape: {self.df.shape}")
        except FileNotFoundError:
            print(f"Error: Dataset not found at {self.data_path}. Please ensure the file exists.")
            raise
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            raise

    def clean_data(self):
        """
        Performs initial data cleaning steps:
        - Renames 'customerID' column.
        - Converts 'TotalCharges' to numeric, handling missing values by coercion.
        - Fills missing 'TotalCharges' with the median.
        - Replaces 'No internet service' and 'No phone service' with 'No' in relevant columns.
        - Maps 'Yes'/'No' target variable to 1/0.
        """
        if self.df is None:
            print("Error: Data not loaded. Call load_data() first.")
            return

        print("Starting data cleaning...")

        # Drop 'customerID' column as it's not useful for prediction
        if 'customerID' in self.df.columns:
            self.df = self.df.drop('customerID', axis=1)
            print("Dropped 'customerID' column.")

        # Convert 'TotalCharges' to numeric, coercing errors to NaN
        # This is important because 'TotalCharges' contains some spaces which become NaN
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        print("Converted 'TotalCharges' to numeric.")

        # Fill missing 'TotalCharges' with the median (or 0, depending on strategy)
        # Missing values in TotalCharges generally correspond to new customers who haven't been charged yet
        median_total_charges = self.df['TotalCharges'].median()
        self.df['TotalCharges'].fillna(median_total_charges, inplace=True)
        print(f"Filled missing 'TotalCharges' with median: {median_total_charges}")

        # Replace 'No internet service' with 'No' for consistency in relevant columns
        internet_service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                 'TechSupport', 'StreamingTV', 'StreamingMovies']
        for col in internet_service_cols:
            self.df[col] = self.df[col].replace('No internet service', 'No')
        print("Replaced 'No internet service' with 'No'.")

        # Replace 'No phone service' with 'No' in 'MultipleLines'
        self.df['MultipleLines'] = self.df['MultipleLines'].replace('No phone service', 'No')
        print("Replaced 'No phone service' with 'No' in 'MultipleLines'.")

        # Map 'Yes'/'No' in 'Churn' target variable to 1/0
        self.df['Churn'] = self.df['Churn'].map({'Yes': 1, 'No': 0})
        print("Mapped 'Churn' target variable to 1/0.")

        print("Data cleaning complete.")

    def preprocess_data(self):
        """
        Applies preprocessing steps including one-hot encoding for categorical features
        and standard scaling for numerical features.
        Sets up the ColumnTransformer for consistent preprocessing.
        """
        if self.df is None:
            print("Error: Data not loaded. Call load_data() first.")
            return

        print("Starting data preprocessing...")

        # Separate features (X) and target (y)
        self.X = self.df.drop('Churn', axis=1)
        self.y = self.df['Churn']

        # Identify categorical and numerical features
        self.numerical_features = self.X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_features = self.X.select_dtypes(include='object').columns.tolist()

        # Create preprocessing pipelines for numerical and categorical features
        numerical_transformer = StandardScaler() # Scale numerical features
        categorical_transformer = OneHotEncoder(handle_unknown='ignore') # One-hot encode categorical features

        # Create a preprocessor using ColumnTransformer
        # This allows applying different transformations to different columns
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='passthrough' # Keep other columns (if any)
        )
        print("Preprocessing pipeline (ColumnTransformer) created.")

    def apply_preprocessing(self, X_data):
        """
        Applies the defined preprocessing steps to the given data.

        Args:
            X_data (pd.DataFrame): The feature DataFrame to preprocess.

        Returns:
            np.ndarray: The preprocessed feature array.
        """
        if self.preprocessor is None:
            print("Error: Preprocessor not set up. Call preprocess_data() first.")
            return None
        print("Applying preprocessing to data...")
        return self.preprocessor.fit_transform(X_data) # Use fit_transform on training data

    def get_feature_names_out(self):
        """
        Returns the names of the features after one-hot encoding.
        This is useful for understanding the columns after preprocessing.

        Returns:
            list: A list of feature names.
        """
        if self.preprocessor is None:
            print("Error: Preprocessor not set up. Call preprocess_data() first.")
            return None
        # Get feature names from the ColumnTransformer after fitting
        return self.preprocessor.get_feature_names_out()


    def split_data(self, test_size=0.2, random_state=42, apply_smote=True):
        """
        Splits the data into training and testing sets and optionally applies SMOTE.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): Controls the shuffling applied to the data before applying the split.
            apply_smote (bool): Whether to apply SMOTE to balance the training data.

        Returns:
            tuple: X_train_processed, X_test_processed, y_train, y_test
        """
        if self.X is None or self.y is None:
            print("Error: Features and target not set. Call preprocess_data() first.")
            return None, None, None, None

        print(f"Splitting data into training ({1-test_size:.0%} and testing ({test_size:.0%}) sets.")
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )

        # Apply preprocessing
        # It's crucial to fit the preprocessor ONLY on the training data
        # and then transform both training and test data.
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test) # Only transform X_test

        print(f"Original training set shape: {X_train_processed.shape}, target shape: {y_train.shape}")
        print(f"Original testing set shape: {X_test_processed.shape}, target shape: {y_test.shape}")

        if apply_smote:
            print("Applying SMOTE to balance the training data...")
            smote = SMOTE(random_state=random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
            print(f"Training data resampled with SMOTE. New shape: {X_train_resampled.shape}, target shape: {y_train_resampled.shape}")
            return X_train_resampled, X_test_processed, y_train_resampled, y_test
        else:
            print("SMOTE not applied.")
            return X_train_processed, X_test_processed, y_train, y_test

    def get_raw_data(self):
        """
        Returns the raw DataFrame after initial loading.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        return self.df

    def get_features_target(self):
        """
        Returns the feature DataFrame (X) and target Series (y) after cleaning.

        Returns:
            tuple: (pd.DataFrame X, pd.Series y)
        """
        return self.X, self.y

    def get_preprocessor(self):
        """
        Returns the fitted ColumnTransformer preprocessor.

        Returns:
            ColumnTransformer: The fitted preprocessor.
        """
        return self.preprocessor

if __name__ == '__main__':
    # Example usage:
    # This block runs only when the script is executed directly
    processor = DataProcessor(data_path='../data/telco_customer_churn.csv')
    processor.load_data()
    processor.clean_data()
    processor.preprocess_data() # Setup the preprocessor

    X_train, X_test, y_train, y_test = processor.split_data(apply_smote=True)

    if X_train is not None:
        print("\nData Processing Complete:")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")

        # Check class distribution in resampled training data
        print(f"Y_train (resampled) value counts:\n{y_train.value_counts()}")
        print(f"Y_test value counts:\n{y_test.value_counts()}")

        # Get the feature names after preprocessing
        feature_names = processor.get_feature_names_out()
        if feature_names is not None:
            print(f"Number of processed features: {len(feature_names)}")
            # print("First 10 processed feature names:", feature_names[:10])
