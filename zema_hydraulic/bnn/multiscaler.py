from sklearn.preprocessing import StandardScaler
import copy

class MultiScaler:
    def __init__(self, scaler_class=StandardScaler):
        self.scaler_class = scaler_class
        self.scalers=[]

    def fit(self,X):
        #expecting 3D array
        self.scalers =[]
        self.num_scalers = X.shape[-1]
        for i in range(self.num_scalers):
            scaler = self.scaler_class()
            scaler.fit(X[:,:,i])
            self.scalers.append(scaler)
        return self

    def transform(self,X):
        #expecting 3D array
        self.num_scalers = len(self.scalers)
        for i in range(self.num_scalers):
            X[:,:,i]=self.scalers[i].transform(X[:,:,i])
        return X

    def fit_transform(self,X):
        #expecting 3D array
        self.scalers =[]
        self.num_scalers = X.shape[-1]
        for i in range(self.num_scalers):
            scaler = self.scaler_class()
            X[:,:,i]=scaler.fit_transform(X[:,:,i])
            self.scalers.append(scaler)
        return X

    def inverse_transform(self,X):
        #expecting 3D array
        temp_X = copy.copy(X)
        self.num_scalers = len(self.scalers)
        for i in range(self.num_scalers):
            temp_X[:,:,i]=self.scalers[i].inverse_transform(temp_X[:,:,i])
        return temp_X

    def inverse_variance(self,X):
        #expecting 3D array
        temp_X = copy.copy(X)
        self.num_scalers = len(self.scalers)
        for i in range(self.num_scalers):
            for sample in range(temp_X.shape[0]):
                temp_X[sample,:,i]=self.scalers[i].scale_*temp_X[sample,:,i]
        return temp_X
