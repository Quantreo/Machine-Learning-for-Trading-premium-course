"""
Description:  Trading strategy based on a Machine Learning algorithm (SVM classifier). As input, we have
              SMA diff, RSI and ATR to have more information.

              We standardize the data to put all the data at the same scale (necessary for many algorithms and
              allow algorithms to have a better convergence, especially SVM)

Entry signal: We need that the ML algo say to buy in the same time

Exit signal:  Basic Take-profit and Stop-loss

Good to know: Only one trade at time (we can't have a buy and a sell position in the same time)

How to improve this algorithm?: Try a non-linear model to see the difference of performances
"""

from Quantreo.DataPreprocessing import *
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, VotingRegressor



class MlCourseStrategy1:

    def __init__(self, data, parameters):
        # Set parameters
        self.list_X = parameters["list_X"]
        self.tp, self.sl = parameters["tp"], parameters["sl"]
        self.cost, self.leverage = parameters["cost"], parameters["leverage"]
        self.train_mode = parameters["train_mode"]
        self.training_data = parameters["training_data"]

        self.model, self.sc = None, None
        self.saved_model_path, self.saved_sc_path = None, None

        # Get test parameters
        self.output_dictionary = parameters.copy()
        self.output_dictionary["train_mode"] = False

        if self.train_mode:
            self.data_train = data
            self.data = data
            self.train_model()
        else:
            self.model = parameters["model"]
            self.sc = parameters["sc"]
            self.data = data

        self.start_date_backtest = self.data.index[0]
        self.get_predictions()

        # Get Entry parameters
        self.buy, self.sell = False, False
        self.open_buy_price, self.open_sell_price = None, None
        self.entry_time, self.exit_time = None, None

        # Get exit parameters
        self.var_buy_high, self.var_sell_high = None, None
        self.var_buy_low, self.var_sell_low = None, None


    def get_features(self, data_sample):
        pass


    def train_model(self):
        # Extract the training data for the train period ONLY
        start_date, end_date = self.data_train.index[0], self.data_train.index[-1]
        model_training_data = self.training_data.loc[start_date:end_date]

        X_train = model_training_data[self.list_X]
        y_train = model_training_data["dummy"]

        # Initialize the standardization model
        sc = StandardScaler()
        X_train_sc = sc.fit_transform(X_train)

        # Create the model
        ml_model = VotingClassifier(
    estimators=[('lr', LogisticRegression()),
                ('linsvc', LinearSVC(max_iter=10_000)),
                ('svc', SVC(C=3)),
                ('rfc', RandomForestClassifier(n_estimators=1_000, max_depth=None, random_state=56)),
               ('mlp', MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100, 20), random_state=56)),
               ('bagsvc',BaggingClassifier(estimator=SVC(C=3),n_estimators=10, random_state=56))], voting='hard', n_jobs=-1)
        ml_model.fit(X_train_sc, y_train)

        # Save models as attributes
        self.model = ml_model
        self.sc = sc

        self.output_dictionary["model"] = ml_model
        self.output_dictionary["sc"] = sc

    def get_predictions(self):

        X = self.data[self.list_X]
        X_sc = self.sc.transform(X)

        predict_array = self.model.predict(X_sc)
        self.data["ml_signal"] = 0
        self.data["ml_signal"] = predict_array

    def get_entry_signal(self, time):
        """
        Random Entry signal
        :param i: row number
        :return: Open a buy or sell position using a random signal
        """
        if time not in self.data.index:
            return 0, self.entry_time

        if len(self.data.loc[:time]["ml_signal"]) < 2:
            return 0, self.entry_time

        # Create entry signal --> -1,0,1
        entry_signal = 0
        if self.data.loc[:time]["ml_signal"][-2] == 1:
            entry_signal = 1
        elif self.data.loc[:time]["ml_signal"][-2] == -1:
            entry_signal = -1

        # Enter in buy position only if we want to, and we aren't already
        if entry_signal == 1 and not self.buy and not self.sell:
            self.buy = True
            self.open_buy_price = self.data.loc[time]["open"]
            self.entry_time = time

        # Enter in sell position only if we want to, and we aren't already
        elif entry_signal == -1 and not self.sell and not self.buy:
            self.sell = True
            self.open_sell_price = self.data.loc[time]["open"]
            self.entry_time = time

        else:
            entry_signal = 0

        return entry_signal, self.entry_time

    def get_exit_signal(self, time):
        """
        Take-profit & Stop-loss exit signal
        :param i: row number
        :return: P&L of the position IF we close it

        **ATTENTION**: If you allow your bot to take a buy and a sell position in the same time,
        you need to return 2 values position_return_buy AND position_return_sell
        """
        # Verify if we need to close a position and update the variations IF we are in a buy position
        if self.buy:
            self.var_buy_high = (self.data.loc[time]["high"] - self.open_buy_price) / self.open_buy_price
            self.var_buy_low = (self.data.loc[time]["low"] - self.open_buy_price) / self.open_buy_price

            # Let's check if AT LEAST one of our threshold are touched on this row
            if (self.tp < self.var_buy_high) and (self.var_buy_low < self.sl):

                # Close with a positive P&L if high_time is before low_time
                if self.data.loc[time]["high_time"] < self.data.loc[time]["low_time"]:
                    self.buy = False
                    self.open_buy_price = None
                    position_return_buy = (self.tp - self.cost) * self.leverage
                    self.exit_time = time
                    return position_return_buy, self.exit_time

                # Close with a negative P&L if low_time is before high_time
                elif self.data.loc[time]["low_time"] < self.data.loc[time]["high_time"]:
                    self.buy = False
                    self.open_buy_price = None
                    position_return_buy = (self.sl - self.cost) * self.leverage
                    self.exit_time = time
                    return position_return_buy, self.exit_time

                else:
                    self.buy = False
                    self.open_buy_price = None
                    position_return_buy = 0
                    self.exit_time = time
                    return position_return_buy, self.exit_time

            elif self.tp < self.var_buy_high:
                self.buy = False
                self.open_buy_price = None
                position_return_buy = (self.tp - self.cost) * self.leverage
                self.exit_time = time
                return position_return_buy, self.exit_time

            # Close with a negative P&L if low_time is before high_time
            elif self.var_buy_low < self.sl:
                self.buy = False
                self.open_buy_price = None
                position_return_buy = (self.sl - self.cost) * self.leverage
                self.exit_time = time
                return position_return_buy, self.exit_time

        # Verify if we need to close a position and update the variations IF we are in a sell position
        if self.sell:
            self.var_sell_high = -(self.data.loc[time]["high"] - self.open_sell_price) / self.open_sell_price
            self.var_sell_low = -(self.data.loc[time]["low"] - self.open_sell_price) / self.open_sell_price

            # Let's check if AT LEAST one of our threshold are touched on this row
            if (self.tp < self.var_sell_low) and (self.var_sell_high < self.sl):

                # Close with a positive P&L if high_time is before low_time
                if self.data.loc[time]["low_time"] < self.data.loc[time]["high_time"]:
                    self.sell = False
                    self.open_sell_price = None
                    position_return_sell = (self.tp - self.cost) * self.leverage
                    self.exit_time = time
                    return position_return_sell, self.exit_time

                # Close with a negative P&L if low_time is before high_time
                elif self.data.loc[time]["high_time"] < self.data.loc[time]["low_time"]:
                    self.sell = False
                    self.open_sell_price = None
                    position_return_sell = (self.sl - self.cost) * self.leverage
                    self.exit_time = time
                    return position_return_sell, self.exit_time

                else:
                    self.sell = False
                    self.open_sell_price = None
                    position_return_sell = 0
                    self.exit_time = time
                    return position_return_sell, self.exit_time

            # Close with a positive P&L if high_time is before low_time
            elif self.tp < self.var_sell_low:
                self.sell = False
                self.open_sell_price = None
                position_return_sell = (self.tp - self.cost) * self.leverage
                self.exit_time = time
                return position_return_sell, self.exit_time

            # Close with a negative P&L if low_time is before high_time
            elif self.var_sell_high < self.sl:
                self.sell = False
                self.open_sell_price = None
                position_return_sell = (self.sl - self.cost) * self.leverage
                self.exit_time = time
                return position_return_sell, self.exit_time

        return 0, None