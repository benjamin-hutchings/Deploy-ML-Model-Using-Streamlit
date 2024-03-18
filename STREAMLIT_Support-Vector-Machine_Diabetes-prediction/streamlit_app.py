import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('./trained_model.sav', 'rb'))

input_data = (9,171,110,24,240,45.4,0.721,54)

def diabetes_prediction(input_data): 
    # convert to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape so we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==0):
        return 'Not Diabetic'
    else:
        return 'Diabetic'
    
def main():
    # giving a title
    st.title('Diabetes Prediction Web App')
    
    # getting input data from the user
    Pregnancies = st.text_input('Number of pregancies: ')
    Glucose = st.text_input('Glucose: ')
    BloodPressure = st.text_input('Blood Pressure: ')
    SkinThickness = st.text_input('Skin Thickness: ')
    Insulin = st.text_input('Insulin Level: ')
    BMI = st.text_input('BMI: ')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function: ')
    Age = st.text_input('Age: ')
    
    # code for prediction
    diagnosis = ''
    
    if st.button('Diagnosis Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,
                                         Glucose,
                                         BloodPressure,
                                         SkinThickness,
                                         Insulin,
                                         BMI,
                                         DiabetesPedigreeFunction,
                                         Age])
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()
        