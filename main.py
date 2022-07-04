import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# importing data and viewing it
wine_data = pd.read_csv("winequality-red.csv")
# print(wine_data.describe())


# finding correlations
corr = wine_data.corr()
# print(corr['quality'].sort_values(ascending=False))
wine_data.drop(["residual sugar", "free sulfur dioxide", "pH"], axis=1, inplace=True)
attributes = ["alcohol", "volatile acidity", "quality"]
# pd.plotting.scatter_matrix(wine_data[attributes],figsize=(20,15))
# plt.show()
# wine_data.plot(x="alcohol",y="quality",kind="scatter",alpha=0.4)
# plt.show()

# train test splitting

train_set, test_set = train_test_split(wine_data, test_size=0.2, random_state=42)
wine_data = train_set.copy()


# creating a pipleline

my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scalar',StandardScaler())
])

# splitting data and labels
wine_labels=wine_data["quality"].copy()
wine_data=wine_data.drop("quality",axis=1)
wine_tr=my_pipeline.fit_transform(wine_data)
# print(wine_tr.shape)

# selecting model
model=RandomForestRegressor()
model.fit(wine_tr,wine_labels)

# testing against some test data
some_data=wine_data[:6]
some_labels=wine_labels[:6]
data_prepared=my_pipeline.fit_transform(some_data)

pred=model.predict(data_prepared)
mse=mean_squared_error(some_labels,pred)
rmse=np.sqrt(mse)
# print(rmse)

# testing against test data
x_test=test_set.drop("quality",axis=1)
y_test=test_set["quality"].copy()
x_test_prepared=my_pipeline.fit_transform(x_test)

final_pred=model.predict(x_test_prepared)
final_pred=np.round(final_pred)
final_mse=mean_squared_error(final_pred,y_test)
final_rmse=np.sqrt(final_mse)
print(list(y_test),final_pred)



# using the model
features=np.array([[9.4,0.27,0.53,0.074,18,0.9962,1.13,12]])
predict=model.predict(features)
print(np.round(predict))




