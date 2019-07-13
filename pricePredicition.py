import pandas as pd
import numpy as np
import seaborn as sns
import operator
from IPython.display import display
from IPython.html import widgets 
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
%matplotlib inline

! pip install ipywidgets

#import clean data

csv_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
df = pd.read_csv(csv_path, header = None,sep=',')

#set header titles
headers = ['symbolizing', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors','body-style',
           'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower',
           'peak-rpm','city-mpg', 'highway-mpg', 'price']
df.columns = headers


 #Cleaning

df.replace({'?':np.nan},regex=False,inplace = True)
df.dropna(subset=['price'], axis =0, inplace= True)
df.dropna(subset=['peak-rpm'], axis =0, inplace= True)
df.dropna(subset=['horsepower'], axis =0, inplace= True)
df.dropna(subset=['curb-weight'], axis =0, inplace= True)
df.dropna(subset=['engine-size'], axis =0, inplace= True)
df.dropna(subset=['highway-mpg'], axis =0, inplace= True)
df.dropna(subset=['normalized-losses'], axis =0, inplace= True)
print(df.head())

#Formatting
df['city-mpg'] = 235/df['city-mpg']
df.rename(columns={'city-mpg': 'city-L/100km'}, inplace=True)
df['price'] = df['price'].astype('int32')
df['horsepower'] = df['horsepower'].astype('int32')

#Plotting
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width,height))
    
    ax1 = sns.distplot(RedFunction, hist=False, color='r',label = RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)
    
    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()
    
def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()

#Training and Testing
y_data = df['price']
x_data = df.drop('price',axis=1) #axis=1 means delete column, axis-0 means delete row
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size=0.15,random_state=1)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train[['horsepower']],y_train)

#Getting R^2 value to determine model evaluation
print("R^2 value for training data is: " + str(lr.score(x_train[['horsepower']], y_train)))


#Multiple Linear Regression

lr = LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)

yhat_train = lr.predict(x_train[['horsepower','curb-weight', 'engine-size', 'highway-mpg']])
yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

Title = "Precited vs Actual Data used to Train Model"
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)

Title = "Predicted vs Actual Data used in Testing"
DistributionPlot(y_test, yhat_test, "Actual Values (Train)", "Predicted Values (Train)", Title)


#Polynomial Regression to obtain better fit between predicted and actual data

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size= 0.45, random_state=0)
pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])

poly = LinearRegression()
poly.fit(x_train_pr, y_train)

yhat = poly.predict(x_test_pr)
yhat[0:5]

print('Predicted Values with degree 5: ', yhat[0:4])
print('Actual Values with degree 5: ', y_test[0:4].values)

#plot polynomial model
PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly,pr) 

#R^2 values of polynomial model
print("R^2 value is: ", poly.score(x_train_pr, y_train))
print("R^2 value is: ", poly.score(x_test_pr, y_test))

#Obtaining best R^2 value between horsepower and price
Rsqu_test = []
order = [1,2,3,4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    lr.fit(x_train_pr, y_train) #create model
    Rsqu_test.append(lr.score(x_test_pr, y_test)) #applying model to test data, output R^2 value


print(Rsqu_test[0:]) #best order is degree 1

#Ridge Regression to adjust model for multi-attribtue model
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg', 'normalized-losses']])
x_test_pr = pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg', 'normalized-losses']])

RidgeModel = Ridge(alpha=0.1)
RidgeModel.fit(x_train_pr, y_train)

yhat = RidgeModel.predict(x_test_pr)

#comparing values of predicted and actual
print("predicted: ",yhat[0:4])
print('Actual:', y_test[0:4].values)


#Selecting Best Alpha value
Rsqu_test = {}
Rsqu_train ={}
dummy1 = []
ALFA = 10 * np.array(range(0,1000))
for alfa in ALFA:
    RidgeModel = Ridge(alpha = alfa)
    RidgeModel.fit(x_train_pr, y_train)
    Rsqu_train[alfa] = RidgeModel.score(x_train_pr,y_train)
    Rsqu_test[alfa] = RidgeModel.score(x_test_pr,y_test)
    #Rsqu_train.append(RidgeModel.score(x_train_pr,y_train))
    #Rsqu_test.append(RidgeModel.score(x_test_pr,y_test))


print(max(Rsqu_test.items(), key=operator.itemgetter(1))[0]) #max R^2 is at alpha = 220


#Final Model:
RidgeModel = Ridge(alpha = 220)
RidgeModel.fit(x_train_pr, y_train)





