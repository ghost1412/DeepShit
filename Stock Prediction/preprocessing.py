import numpy as np 


# FUNCTION TO CREATE 1D DATA INTO TIME SERIES DATASET
def data_process(scaled_data, date_range):
    X=[]
    Y=[]
    num_data=len(scaled_data)
    for i in range(date_range, num_data):
        X.append(scaled_data[i-date_range:i, 0])
        Y.append(scaled_data[i, 0])
    
    return  np.array(X), np.array(Y)

# THIS FUNCTION CAN BE USED TO CREATE A TIME SERIES DATASET FROM ANY 1D ARRAY	
