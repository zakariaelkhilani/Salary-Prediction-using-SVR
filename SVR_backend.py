
def svr(file_name,kernel_function_p,c_p,gamma_p,degree_p,eps):
    # Importing the libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    # import OneHotEncoder
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    import matplotlib.pyplot as plt
    from sklearn.svm import SVR

    
    # ==================================================================================================
    #                      Import  DATASET 
    # ==================================================================================================
    
    #file_name="C:/Users/ELKHILANI/Desktop/Time series project/SVR/Position_Salaries.csv"
    
    dataset = pd.read_csv(file_name)
    X = dataset.iloc[:, 1:2].values
    y = dataset.iloc[:, 2].values
    
    
    X = X.reshape(-1,1)
    y = y.reshape(-1,1)
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y)
    
    X = X.reshape(-1,1)
    y = y.reshape(-1,1)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)
    
    # Fitting SVR to the dataset
    
    if kernel_function_p=='poly':
        
        model=SVR(kernel = kernel_function_p,C=c_p,degree=degree_p,gamma=gamma_p,epsilon=eps)
        
    elif  kernel_function_p=='rbf':
        
        model=SVR(kernel = kernel_function_p,C=c_p,gamma=gamma_p,epsilon=eps)
        
    elif  kernel_function_p=='sigmoid':
        model=SVR(kernel = kernel_function_p,C=c_p,gamma=gamma_p,epsilon=eps)
        
    elif  kernel_function_p=='linear':
        model=SVR(kernel = kernel_function_p,C=c_p,epsilon=eps)


    model.fit(X_train, y_train)
    
    
    ## PRedict 
    
    y_pred = model.predict(X_test)
    
    ## Inverse
    
    Y_pred_inversed=sc_y.inverse_transform(y_pred)
    Y_test_inversed=sc_y.inverse_transform(y_test)
    X_test_inversed=sc_y.inverse_transform(X_test)

    
    mape=np.mean( np.abs((Y_test_inversed - Y_pred_inversed) / Y_test_inversed)) * 100
    
    return Y_pred_inversed,Y_test_inversed,X_test_inversed,mape
    