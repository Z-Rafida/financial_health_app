import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'sme_financial_health_train.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')

TARGET_COL   = 'Target'
TEST_SIZE    = 0.2
RANDOM_STATE = 42

DROP_COLS = ['ID']

CATEGORICAL_COLS = [
    'country', 'owner_sex',
    'attitude_stable_business_environment', 'attitude_worried_shutdown',
    'compliance_income_tax', 'perception_insurance_doesnt_cover_losses',
    'perception_cannot_afford_insurance', 'motor_vehicle_insurance',
    'has_mobile_money', 'current_problem_cash_flow', 'has_cellphone',
    'offers_credit_to_customers', 'attitude_satisfied_with_achievement',
    'has_credit_card', 'keeps_financial_records',
    'perception_insurance_companies_dont_insure_businesses_like_yours',
    'perception_insurance_important', 'has_insurance', 'covid_essential_service',
    'attitude_more_successful_next_year', 'problem_sourcing_money',
    'marketing_word_of_mouth', 'has_loan_account', 'has_internet_banking',
    'has_debit_card', 'future_risk_theft_stock', 'medical_insurance',
    'funeral_insurance', 'motivation_make_more_money',
    'uses_friends_family_savings', 'uses_informal_lender'
]

NUMERICAL_COLS = [
    'owner_age', 'personal_income', 'business_expenses',
    'business_turnover', 'business_age_years', 'business_age_months'
]

TARGET_CLASSES  = ['Low', 'Medium', 'High']
TARGET_MAPPING  = {'Low': 0, 'Medium': 1, 'High': 2}

MODEL_PARAMS = {
    'n_estimators':  200,
    'learning_rate': 0.05,
    'num_leaves':    31,
    'random_state':  RANDOM_STATE,
    'verbose':       -1
}
