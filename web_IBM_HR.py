import streamlit as st
import pandas as pd
from joblib import load

model = load('./model/model_hr.joblib')


# función para clasificar
def classify(attrition):
    if attrition == 'No':
        return 'No, employee WITH attrition'
    elif attrition == 'Yes':
        return 'Yes, employee WITHOUT attrition'

def main():
    st.title("IBM HR Analytics Employee Attrition & Performance")

    st.sidebar.header('User Input Parameters')
    
    def user_input_parameters():
        monthlyIncome = st.sidebar.slider('1.- Monthly Income ($)', 1000, 6000, 20000)
        age = st.sidebar.slider('2.- Age', 18, 70, 35)
        totalWorkingYears = st.sidebar.slider('3.- Total Working Years', 0, 11, 40)
        overTime = st.sidebar.selectbox('4.- Over Time', [0, 1])
        distanceFromHome = st.sidebar.slider('5.- Distance From Home (Km)', 1, 10, 30)
        dailyRate = st.sidebar.slider('6.- Daily Rate', 103, 795, 1500)
        monthlyRate = st.sidebar.slider('7.- Monthly Rate', 2000, 14250, 27000)
        hourlyRate = st.sidebar.slider('8.- Hourly Rate', 30, 68, 100)
        yearsAtCompany = st.sidebar.slider('9.- Years At Company', 0, 7, 40)
        numCompaniesWorked = st.sidebar.slider('10.- Number of Companies Worked', 0, 3, 10)

        data = {
                'age': age,
                'monthlyincome': monthlyIncome,
                'totalWorkingYears' : totalWorkingYears,
                'overTime' : overTime,
                'distanceFromHome' : distanceFromHome,
                'dailyRate' : dailyRate,
                'monthlyRate' : monthlyRate,
                'hourlyRate' : hourlyRate,
                'yearsAtCompany' : yearsAtCompany,
                'numCompaniesWorked' : numCompaniesWorked
                }
                
        features = pd.DataFrame(data, index=[0])
        return features
    
    df = user_input_parameters()

    st.subheader('User Input Parameters')
    st.subheader(model)
    st.write(df)

    if st.button('RUN'):
        st.success(classify(model.predict(df)))

if __name__ == '__main__':
    main()