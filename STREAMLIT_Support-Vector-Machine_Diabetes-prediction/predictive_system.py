import numpy as np
import pickle

input_data = (9,171,110,24,240,45.4,0.721,54)

loaded_model = pickle.load(open('./trained_model.sav', 'rb'))

# convert to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape so we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)

print(prediction)

if (prediction[0]==0):
    print('Not Diabetic')
else:
    print('Diabetic')