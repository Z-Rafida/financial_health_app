import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.predict import load_model, predict_single, predict_batch
from src.config import MODEL_PATH, TARGET_CLASSES

st.set_page_config(page_title='SME Financial Health Predictor', layout='wide')
st.title('SME Financial Health Predictor')
st.markdown('Predict the financial health category of a small business across Southern Africa.')

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error('Model not found. Run python src/train.py first.')
        st.stop()
    return load_model()

model = get_model()
tab1, tab2 = st.tabs(['Single Business', 'Batch Prediction'])

with tab1:
    st.header('Single Business Assessment')
    col1, col2, col3 = st.columns(3)
    with col1:
        country        = st.selectbox('Country', ['eswatini', 'lesotho', 'zimbabwe', 'malawi'])
        owner_age      = st.number_input('Owner Age', 18, 80, 35)
        owner_sex      = st.selectbox('Owner Gender', ['Male', 'Female'])
        business_age_years = st.number_input('Business Age (Years)', 0, 50, 5)
        business_turnover  = st.number_input('Monthly Turnover', 0.0, 1000000.0, 5000.0)
        business_expenses  = st.number_input('Monthly Expenses', 0.0, 500000.0, 3000.0)
        personal_income    = st.number_input('Personal Income', 0.0, 500000.0, 4000.0)
    with col2:
        has_mobile_money = st.selectbox('Has Mobile Money?', ['Yes', 'No'])
        has_cellphone    = st.selectbox('Has Cellphone?', ['Yes', 'No'])
        has_credit_card  = st.selectbox('Has Credit Card?', ['Yes', 'No'])
        has_loan_account = st.selectbox('Has Loan Account?', ['Yes', 'No'])
        has_insurance    = st.selectbox('Has Insurance?', ['Yes', 'No'])
        keeps_financial_records = st.selectbox('Keeps Financial Records?', ['Yes', 'No'])
        compliance_income_tax   = st.selectbox('Pays Income Tax?', ['Yes', 'No'])
    with col3:
        problem_sourcing_money = st.selectbox('Problem Accessing Finance?', ['Yes', 'No'])
        current_problem_cash_flow = st.selectbox('Cash Flow Problem?', ['Yes', 'No'])
        uses_informal_lender   = st.selectbox('Uses Informal Lender?', ['Yes', 'No', 'Never had'])
        attitude_worried_shutdown = st.selectbox('Worried Business Will Close?', ['Yes', 'No'])
        attitude_more_successful_next_year = st.selectbox('Expects More Success Next Year?', ['Yes', 'No'])

    if st.button('Assess Financial Health', type='primary'):
        input_data = {
            'country': country, 'owner_age': owner_age, 'owner_sex': owner_sex,
            'business_age_years': business_age_years,
            'business_age_months': business_age_years * 12,
            'business_turnover': business_turnover, 'business_expenses': business_expenses,
            'personal_income': personal_income, 'has_mobile_money': has_mobile_money,
            'has_cellphone': has_cellphone, 'has_credit_card': has_credit_card,
            'has_loan_account': has_loan_account, 'has_insurance': has_insurance,
            'keeps_financial_records': keeps_financial_records,
            'compliance_income_tax': compliance_income_tax,
            'problem_sourcing_money': problem_sourcing_money,
            'current_problem_cash_flow': current_problem_cash_flow,
            'uses_informal_lender': uses_informal_lender,
            'attitude_worried_shutdown': attitude_worried_shutdown,
            'attitude_more_successful_next_year': attitude_more_successful_next_year,
            'attitude_stable_business_environment': 'Yes',
            'perception_insurance_doesnt_cover_losses': 'No',
            'perception_cannot_afford_insurance': 'No',
            'motor_vehicle_insurance': 'No', 'covid_essential_service': 'No',
            'attitude_satisfied_with_achievement': 'Yes',
            'perception_insurance_companies_dont_insure_businesses_like_yours': 'No',
            'perception_insurance_important': 'Yes', 'offers_credit_to_customers': 'No',
            'attitude_satisfied_with_achievement': 'Yes', 'has_internet_banking': 'No',
            'has_debit_card': 'No', 'future_risk_theft_stock': 'No',
            'medical_insurance': 'No', 'funeral_insurance': 'No',
            'motivation_make_more_money': 'Yes', 'uses_friends_family_savings': 'No',
            'marketing_word_of_mouth': 'Yes'
        }
        result = predict_single(model, input_data)
        st.divider()
        st.metric('Financial Health Category', result['label'])
        color_map = {'Low': 'error', 'Medium': 'warning', 'High': 'success'}
        if result['label'] == 'High':
            st.success('This business has strong financial health.')
        elif result['label'] == 'Medium':
            st.warning('This business has moderate financial health. Some areas need improvement.')
        else:
            st.error('This business has low financial health. Significant support is needed.')

with tab2:
    st.header('Batch Prediction from CSV')
    uploaded = st.file_uploader('Upload CSV', type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        if st.button('Run Predictions', type='primary'):
            results = predict_batch(model, df)
            st.dataframe(results[['Prediction', 'Label']].head(20))
            c1, c2, c3 = st.columns(3)
            c1.metric('Total Businesses', len(results))
            c2.metric('High Financial Health', int((results['Label'] == 'High').sum()))
            c3.metric('Low Financial Health', int((results['Label'] == 'Low').sum()))
            fig, ax = plt.subplots(figsize=(5, 4))
            results['Label'].value_counts().plot(kind='bar', ax=ax,
                color=['tomato', 'steelblue', 'seagreen'], edgecolor='black')
            ax.set_title('Financial Health Distribution')
            ax.tick_params(axis='x', rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.download_button('Download Predictions', results.to_csv(index=False), 'predictions.csv')
