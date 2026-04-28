import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("Loading dataset...")
    # Ensure paths are correct relative to project root
    data_path = 'data/train.csv'
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # Feature Selection for simplicity and real-world UI usage
    # We select features that a user can reasonably input into a web app
    features = [
        'LotArea', 'OverallQual', 'YearBuilt', 'TotalBsmtSF', 
        'GrLivArea', 'FullBath', 'BedroomAbvGr', 'GarageCars'
    ]
    target = 'SalePrice'
    
    # Data Cleaning
    print("Cleaning data...")
    df_selected = df[features + [target]].copy()
    
    # Handle missing values (simple imputation with median for this beginner project)
    df_selected.fillna(df_selected.median(), inplace=True)
    
    # Train-test split
    X = df_selected[features]
    y = df_selected[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Training: Decision Tree vs Random Forest
    print("Training Decision Tree Regressor...")
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    
    print("Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluation
    def evaluate(model, name):
        preds = model.predict(X_test)
        # Convert metrics to INR (1 USD = 83 INR)
        rmse_inr = np.sqrt(mean_squared_error(y_test, preds)) * 83
        mae_inr = mean_absolute_error(y_test, preds) * 83
        r2 = r2_score(y_test, preds)
        print(f"\n--- {name} Performance (in INR) ---")
        print(f"RMSE: ₹{rmse_inr:,.2f}")
        print(f"MAE:  ₹{mae_inr:,.2f}")
        print(f"R²:   {r2:.4f}")
        return preds * 83 # Return predictions in INR for plotting

    dt_preds = evaluate(dt_model, "Decision Tree")
    rf_preds = evaluate(rf_model, "Random Forest")
    
    # We choose Random Forest as it typically performs better
    print("\nSaving Random Forest model to models/rf_model.joblib...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, 'models/rf_model.joblib')
    
    # Save Feature columns for the API
    joblib.dump(features, 'models/features.joblib')
    print("Model saved successfully!")

    # Visualization: Actual vs Predicted
    print("Generating Evaluation Plot...")
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('images', exist_ok=True)
    
    y_test_inr = y_test * 83
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_inr, rf_preds, alpha=0.5, color='blue', label='Predicted vs Actual')
    plt.plot([y_test_inr.min(), y_test_inr.max()], [y_test_inr.min(), y_test_inr.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel("Actual Sale Price (₹)")
    plt.ylabel("Predicted Sale Price (₹)")
    plt.title("Random Forest: Actual vs Predicted House Prices")
    plt.legend()
    plt.grid(True)
    plt.savefig('images/actual_vs_predicted.png')
    plt.close()
    
    # Feature Importance Plot
    plt.figure(figsize=(10, 6))
    importances = rf_model.feature_importances_
    sns.barplot(x=importances, y=features, palette='viridis')
    plt.title('Feature Importances in Random Forest')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('images/feature_importance.png')
    plt.close()
    
    print("Plots saved to images/ directory.")

if __name__ == "__main__":
    main()
