import streamlit as st
import pandas as pd
import joblib

#Load trained model
model=joblib.load('heart_model.pkl')
features=joblib.load('features.pkl')

#page Configurations
st.set_page_config(
    page_title='Heart Disease Prediction',
    page_icon='❤️',
    layout='wide',
)
st.title('❤️ Heart Disease Prediction System')
st.write('Machine Learning Based Health Risk Prediction')
st.warning('Educational demonstration only. '
           'This prediction is not a medical diagnosis or a substitute for '
           'professional medical evaluation')
st.divider()
st.subheader('Enter Patient Details')
col1,col2,col3=st.columns(3)

with col1:
    age=st.number_input('Age',min_value=1,max_value=120,value=45)
    sex=st.selectbox("Sex",[0,1],
                     format_func=lambda x: 'Female' if x==0 else 'Male')
    cp=st.selectbox('Chest Pain Type',[0,1,2,3])
    trestbps=st.number_input('Resting Blood Pressure(mm Hg)',min_value=50,max_value=250,value=120)
    chol=st.number_input('Cholesterol(mg/dl)',min_value=50,max_value=700,value=200)

with col2:
    fbs=st.selectbox('Fasting Blood Sugar<120(mg/dl)',[0,1])
    restecg=st.selectbox('Resting ECG',[0,1,2])
    thalach=st.number_input('Maximum Heart Rate',min_value=40,max_value=250,value=150)
    exang=st.selectbox('Exercise Induced Angina',[0,1])
    oldpeak=st.number_input('Oldpeak',min_value=0.0,max_value=10.0,value=1.0,step=0.1)

with col3:
    slope=st.selectbox('Slope',[0,1,2])
    ca=st.selectbox('Number of Major Vessels',[0,1,2,3,4])
    thal=st.selectbox('Thalassemia',[0,1,2,3])
st.divider()

if st.button('Predict Heart Disease',type='primary'):
    input_data={
        'age':age,
        'sex':sex,
        'cp':cp,
        'trestbps':trestbps,
        'chol':chol,
        'fbs':fbs,
        'restecg':restecg,
        'thalach':thalach,
        'exang':exang,
        'oldpeak':oldpeak,
        'slope':slope,
        'ca':ca,
        'thal':thal
    }
    #Arrange inputs in th training feature order
    input_df=pd.DataFrame([[input_data[col] for col in features]],
                          columns=features)
    prediction=model.predict(input_df)[0]
    st.subheader('Prediction Result')
    if prediction==1:
        st.error('The model predicts a potential heart disease risk. Please consult a Qualified doctor.')
    else:
        st.success('The model did not identify the positive heart disease class for this input')
    st.info('This result is a ML prediction not a clinical assessment. '
            'A negative prediction does not rule out heart disease')
st.divider()
st.caption('Developed using Python,Scikit-Learn and Streamlit')