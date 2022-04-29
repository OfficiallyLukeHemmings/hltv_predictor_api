import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

def training_data_import(filename='training_data_pos.csv'):
    """Importing of the training data and returning of pd.DataFrames (y and X)
    required for training the ML model.

    Args:
        filename (string): Filename of the training_data csv file (defaulting 
                            to 'training_data.csv')
        
    Returns:
        y (pd.DataFrame): 'Game score' values from csv_data
                            'Game score' is a boolean (0/1, with 1 being games
                            that should be recommended)
        X (pd.DataFrame): 'csv data as pd.DataFrame
    """
    # importing csv data
    csv_data = pd.read_csv(filename)
    
    # Popping pd.DataFrame of 'Game Score' values from csv_data
    y = csv_data.pop('gameScore')
    # pd.DataFrame of csv_data 
    X = csv_data
    
    return X, y


class Predictor():
    """_summary_
    """
    def __init__(self, optimistic=True):
        """_summary_
        """
        self.clf = GradientBoostingClassifier(n_estimators=100,
                                              learning_rate=1.0, max_depth=1)
        self.optimistic = optimistic
    
    def train(self):
        """_summary_
        """
        if self.optimistic:
            X_train, y_train = training_data_import()
        else:
            X_train, y_train = training_data_import('training_data_neg.csv')
        
        self.clf.fit(X_train, y_train)
        
    def pred(self, data):
        """_summary_

        Args:
            data (_type_): _description_json

        Returns:
            _type_: _description_
        """
        return self.clf.predict(data).item()