import pandas as pd
from sklearn.linear_model import LogisticRegression

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
    """A Machine Learning Model class using sckit-learn's Logistic Regression 
    algorithm (actually a classifier in sklearn). 
    
    Attributes:
        clf = The base logistic regression model for use in this class' 
            methods (train and pred)
        optimistic = Bool indicating whether the class instance should take
            an optimistic or pessimistic approach to evaluating games.
            Implemented by training using  the corresponding training data.
            Optimistic training data, for the purposes of training, has merged
            neutral response scores with positive response scores - i.e. in
            the hopes that neutral games are ones to be enjoyed. While the
            pessimistic training data has merged neutral scores with negatives
            instead, taking a more reserved approach for predictions.
    """
    def __init__(self, optimistic=True):
        """Optional Arg: 
            optimistic = Bool (defaulting to true)
        """
        self.clf = LogisticRegression()
        self.optimistic = optimistic
    
    def train(self):
        """Class method used for training the ML model. The training data used
        is dependent on the optimistic attribute (bool), allowing the model to 
        take either a pessimistic or optimistic approach to predictions.
        """
        # Preparing the data for training
        # X_train and y_train = Pandas DataFrames
        if self.optimistic:
            X_train, y_train = training_data_import()
        else:
            X_train, y_train = training_data_import('training_data_neg.csv')
        
        # Training the model
        self.clf.fit(X_train, y_train)
        
    def pred(self, data):
        """Class method for making predictions given a DataFrame outline of the 
        game to be evaluated.
            
        Args:
            data (Pandas DataFrame): ...of the following: 
                - eventType (Int): Where... 0=Online, 1=LAN, 2=Major event
                - team1Rank (Int): Team 1's World Ranking
                - team2Rank (Int): Team 2's World Ranking 
                - bettingOddsDiff (Float): Float of average betting odds 
                    difference as per the HLTV match page. 

        Returns:
            Predicted Enjoyability Value: (0=Not predicted to enjoy, 
                                            1=Predicted to enjoy)
        """
        return self.clf.predict(data).item()