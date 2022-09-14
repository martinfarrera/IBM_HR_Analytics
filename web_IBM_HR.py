import streamlit as st
import pandas as pd
from joblib import load

model = load('./model/model_hr.joblib')


# función para clasificar
def classify(attrition):
    if attrition == 0:
        return 'No, employee WITHOUT attrition'
    elif attrition == 1:
        return 'Yes, employee WITH attrition'

def main():
    st.title("IBM HR Analytics Employee Attrition")
    st.sidebar.header('User Input Parameters')
    
    def user_input_parameters():
        stockOptionLevel = st.sidebar.selectbox('1.- Stock Option Level', [0, 1, 2, 3])
        monthlyIncome = st.sidebar.slider('2.- Monthly Income', 1091, 5574, 19999)
        departmentRD = st.sidebar.selectbox('3.- Department_Research & Development',[ 0, 1])
        jobSatisfaction = st.sidebar.selectbox('4.- Job Satisfaction', [1, 3, 2, 4])
        maritalStatus = st.sidebar.selectbox('5.- Marital Status', [0, 1])
        yearsInCurrentRole = st.sidebar.slider('6.- Years In Current Role', 0, 3, 18)
        age = st.sidebar.slider('7.- Age', 18, 35, 60)
        totalWorkingYears = st.sidebar.slider('8.- Total Working Years', 0, 10, 40)
        monthlyRate = st.sidebar.slider('9.- Monthly Rate', 2094, 14232, 26997)
        hourlyRate = st.sidebar.slider('10.- Hourly Rate', 30, 67, 100)
        distanceFromHome = st.sidebar.slider('11.- Distance From Home (Km)', 1, 10, 29)
        dailyRate = st.sidebar.slider('12.- Daily Rate', 103, 792, 1496)
        environmentSatisfaction = st.sidebar.selectbox('13.- Environment Satisfaction', [1, 2, 3, 4])
        jobInvolvement = st.sidebar.selectbox('14.- Job Involvement', [1, 2, 3, 4])
        yearsWithCurrManager = st.sidebar.slider('15.- Years With Current Manager', 0, 3, 4)
        relationshipSatisfaction = st.sidebar.selectbox('16.- Relationship Satisfaction', [1, 2, 3, 4])
        yearsAtCompany = st.sidebar.slider('17.- Years At Company', 0, 6, 373)
        educationField_Medical = st.sidebar.selectbox('18.- Education Field (Medical)', [0, 1])
        jobLevel = st.sidebar.slider('19.- Job Level', 1, 2, 5)
        workLifeBalance = st.sidebar.selectbox('20.- WorkLife Balance', [1, 2, 3, 4])
        percentSalaryHike = st.sidebar.slider('21.- Percent Salary Hike', 11, 15, 25)
        numCompaniesWorked = st.sidebar.slider('22.- Num Companies Worked', 0, 3, 9)
        trainingTLYear = st.sidebar.slider('23.- Training Times Last Year', 0, 3, 6)
        education = st.sidebar.selectbox('24.- Education Level', [1, 2, 3, 4, 5])
        educationFLS = st.sidebar.selectbox('25.- EducationField (Life Sciences)', [0, 1])

        data = {
                'StockOptionLevel': stockOptionLevel,
                'Monthlyincome': monthlyIncome,
                'DepartmentR&D' : departmentRD,
                'JobSatisfaction' : jobSatisfaction,
                'MaritalStatus' : maritalStatus,
                'YearsInCurrentRole' : yearsInCurrentRole,
                'Age' : age,
                'totalWorkingYears' : totalWorkingYears,
                'MonthlyRate' : monthlyRate,
                'HourlyRate' : hourlyRate,
                'DistanceFromHome' : distanceFromHome,
                'DailyRate' : dailyRate,
                'EnvironmentSatisfaction' : environmentSatisfaction,
                'JobInvolvement' : jobInvolvement,
                'YearsWithCurrManager' : yearsWithCurrManager,
                'RelationshipSatisfaction' : relationshipSatisfaction,
                'YearsAtCompany' : yearsAtCompany,
                'EducationField_Med' : educationField_Medical,
                'JobLevel' : jobLevel,
                'WorkLifeBalance' : workLifeBalance,
                'PercentSalaryHike' : percentSalaryHike,
                'NumCompaniesWorked' : numCompaniesWorked,
                'TrainingTimesLastYear' : trainingTLYear,
                'Education' : education,
                'EducationFLS' : educationFLS,
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