import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class MultiRegressor:
    def __init__(self, model_type='decision_tree', alpha=1.0, degree=2):
        """
        Initializes the model with StandardScaler for feature scaling and selected regression model.
        :param model_type: String representing the type of model to use ('decision_tree', 'random_forest',
                            'svm', 'ridge', 'lasso', 'linear', 'polynomial')
        :param alpha: Regularization strength for Ridge and Lasso
        :param degree: Degree of polynomial features for Polynomial Regression
        """
        self.sc_X = StandardScaler()
        self.sc_y = StandardScaler()
        self.model_type = model_type
        self.degree = degree

        if model_type == 'decision_tree':
            self.model =  DecisionTreeRegressor(
                max_depth=15,
                random_state=42,
                splitter="random")


        elif model_type == 'random_forest':
            self.model =  RandomForestRegressor(
            n_estimators=100,  # Number of trees
            max_depth=30,      # Maximum depth of each tree
            random_state=42,   # Ensures reproducibility
            n_jobs=-1,          # Utilize all processors for training
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=1,
            bootstrap=True
        )

        elif model_type == 'svm':
            self.model = self.svm_rbf=SVR(
                kernel='rbf',
                C=2000,
                gamma=1
            )

        elif model_type == 'ridge':
            self.model = Ridge(alpha=alpha, random_state=42)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=alpha, random_state=42)
        elif model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'polynomial':
            self.poly = PolynomialFeatures(degree=self.degree)
            self.model = LinearRegression()
        else:
            raise ValueError("Invalid model_type. Choose from ['decision_tree', 'random_forest', 'svm', 'ridge', 'lasso', 'linear', 'polynomial']")

    def train(self, X_train, y_train):
        """
        Trains the selected regression model on the scaled data.
        """
        X = self.sc_X.fit_transform(X_train)
        y = self.sc_y.fit_transform(y_train)

        if self.model_type == 'polynomial':
            X_poly = self.poly.fit_transform(X)
            self.model.fit(X_poly, y)
        else:
            self.model.fit(X, y.ravel())

    def predict(self, X_test):
        """
        Predicts the target for given test features after scaling.
        """
        X_test = self.sc_X.transform(X_test)

        if self.model_type == 'polynomial':
            X_test_poly = self.poly.transform(X_test)
            y_pred = self.model.predict(X_test_poly)
        else:
            y_pred = self.model.predict(X_test)

        y_pred = y_pred.reshape(-1, 1)
        self.y_pred = self.sc_y.inverse_transform(y_pred)
        return self.y_pred

    def evaluate(self, y_test,mod):
        """
        Evaluates the model's performance on test data.
        """
        metrics = {
            f"{mod} RMSE": np.sqrt(mean_squared_error(y_test, self.y_pred)),
            f"{mod} R2": r2_score(y_test, self.y_pred)
        }
        self.performance_matrix = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
        return self.performance_matrix

    def R2_mse(self, y_test):
        R2 = r2_score(y_test, self.y_pred)
        MSE = np.sqrt(mean_squared_error(y_test, self.y_pred))
        return R2, MSE

    def plot_performance_matrix(self, performance_matrix):
        fig, ax = plt.subplots()
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=performance_matrix.values, colLabels=performance_matrix.columns, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.auto_set_column_width(col=list(range(len(performance_matrix.columns))))
        plt.show()
    
    # save image
    def save_img(self, plt_obj, name,field):
        """
        Function to save a given plot object to the specified path.
        :param plt_obj: Matplotlib plot object.
        :param save_path: Path to save the plot.
        """
        
        save_path = f'output/{field}/image/{name}.png'
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt_obj.savefig(save_path)
        print(f"Plot saved at {save_path}")


    
