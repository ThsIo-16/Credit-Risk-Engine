"""Machine Learning Pipeline for Credit Risk Prediction"""
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_risk_model(data_path):
    print("Loading Processed Data...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Error: Cleaned data not found. Run data_preprocessing.py first.")
        return None

    print("Preparing features and target variable...")
    # Target: loan_status (1 = Default, 0 = Non-Default)
    y = df['loan_status']
    X = df.drop('loan_status', axis=1)

    #Dummy Var for Categorical Val 
    categorical_columns = X.select_dtypes(include=['object']).columns
    print(f"Found categorical columns to encode: {list(categorical_columns)}")
    X = pd.get_dummies(X, columns=categorical_columns)

    print(f"Dataset shape after encoding categorical variables: {X.shape}")

    # The classic 80% training / 20% test split
    print("Splitting data into Train and Test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    print("\nStarting Hyperparameter Tuning (Testing 10 different blends)...")
    
    # We define a 'grid' of possible settings for the Random Forest
    param_distributions = {
        'n_estimators': [50, 100, 200],       # How many trees
        'max_depth': [5, 10, 15, None],       # How deep the trees can grow
        'min_samples_split': [2, 5, 10]       # Rules for splitting data
    }

    # Base model
    rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')

    # RandomizedSearchCV tests 10 random combinations (n_iter=10) from our grid using Cross-Validation
    rf_random_search = RandomizedSearchCV(
        estimator=rf_base, 
        param_distributions=param_distributions, 
        n_iter=10,            
        cv=3,                  
        scoring='f1',          
        random_state=42, 
        n_jobs=-1
    )

    # Fit the random search
    rf_random_search.fit(X_train, y_train)

    print(f"Best Model Settings Found: {rf_random_search.best_params_}")

    # Extract the winning model
    best_rf_model = rf_random_search.best_estimator_

    print("\n--- Model Evaluation on Unseen Test Data ---")
    y_pred = best_rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
    
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred))

    return best_rf_model, X.columns

if __name__ == "__main__":
    import os
    import joblib  

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data_path = os.path.join(base_dir, 'data', 'processed', 'cleaned_credit_risk.csv')

    # Τρέχουμε την εκπαίδευση
    trained_model, feature_names = train_risk_model(processed_data_path)
    
    if trained_model is not None:
        #Make the path for src
        save_dir = os.path.join(base_dir, 'src')
        
        #Sec Point
        os.makedirs(save_dir, exist_ok=True) 
        joblib.dump(trained_model, os.path.join(save_dir, 'risk_model.pkl'))
        joblib.dump(feature_names, os.path.join(save_dir, 'model_features.pkl'))
        
        print("\nPipeline execution completed. Model saved as .pkl successfully!")
