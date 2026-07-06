import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier
import lightgbm as lgb
from scipy.stats import randint, loguniform, uniform

from preprocessing import load_data, prepare_features
from config import MODEL_PATH, RANDOM_STATE, TARGET_CLASSES, TARGET_MAPPING


def get_models_and_params():
    """
    Define all models. Use class_weight='balanced' for all
    to handle the imbalance (Low=65%, Medium=30%, High=5%).
    """
    models = {
        'Decision Tree': (
            DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE),
            {
                'max_depth':        randint(3, 15),
                'min_samples_leaf': randint(1, 20)
            }
        ),
        # TODO: Add Random Forest with class_weight='balanced'
        'Random Forest': (
            RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
            {
                'n_estimators': randint(100, 500),
                'max_depth': randint(3, 15),
                'min_samples_leaf': randint(1, 20)
            }
        ),
        # TODO: Add XGBoost
        'XGBoost': (
            XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss', n_jobs=-1),
            {
                'n_estimators': randint(100, 500),
                'max_depth': randint(3, 6),
                'learning_rate': [0.01, 0.05, 0.1, 0.2]
            }
        ),
        # TODO: Add LightGBM
        'LightGBM': (
            lgb.LGBMClassifier(class_weight= 'balanced', random_state=RANDOM_STATE, n_jobs=-1, verbose = -1),
            {
               'n_estimators': randint(100, 500),
               'max_depth': randint(3, 7),
               'learning_rate':[0.01, 0.05, 0.1, 0.2]
            }
        ),
        # TODO: Add Logistic Regression
        'Logistics Regression': (
            LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE, max_iter=1000, n_jobs=-1),
            {
                'C': loguniform(0.01, 10)
            }
        ),
    }
    return models


def tune_and_compare(models, X_train, y_train, X_test, y_test, n_iter=20, cv=5):
    """
    Run RandomizedSearchCV. Use scoring='f1_macro'.
    """
    results = []
    best_models = {}
    
    for name, (model, params) in models.items():
        print(f'Tuning {name}...')

        # TODO: RandomizedSearchCV with scoring='f1_macro'
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=n_iter,
            scoring='f1_macro',
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=0
        )

        # TODO: Fit, predict, compute f1_score with average='macro'
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average = 'macro')

        # TODO: Print classification_report with target_names=TARGET_CLASSES
        print('Classification Report')
        print(classification_report(y_test, y_pred, target_names=TARGET_CLASSES, zero_division=0))

        # TODO: Append results and store best model
        results.append({
            'Model': name,
            'Best Params':search.best_params_,
            'CV F1 Macro': search.best_score_,
            'Test F1 Macro': f1
        })
        best_models[name] = best_model
        
    results_df = pd.DataFrame(results).sort_values('Test F1 Macro', ascending=False)
    return results_df, best_models


def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved: {path}')


if __name__ == '__main__':
    df = load_data()
    print(f'Data loaded: {df.shape}')

    X_train, X_test, y_train, y_test = prepare_features(df)
    print(f'Train: {X_train.shape}  |  Test: {X_test.shape}')

    models = get_models_and_params()

    print('\nRunning RandomizedSearchCV...\n')
    results_df, best_models = tune_and_compare(models, X_train, y_train, X_test, y_test)

    print('\n--- Model Comparison ---')
    print(results_df[['Model', 'CV F1 Macro', 'Test F1 Macro']].to_string(index=False))

    best_name  = results_df.iloc[0]['Model']
    best_model = best_models[best_name]
    print(f'\nBest model: {best_name}')

    save_model(best_model)
    print('Training complete.')
