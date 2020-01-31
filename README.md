
# Interactions

## Introduction

In this section, you'll learn about interactions and how to account for them in regression models.

## Objectives

You will be able to:
- Define interactions in a linear model
- Describe how to accommodate for interactions using the parameters of a  linear regression 

## What are interactions?

In statistics, an interaction is a particular property of three or more variables, where two or more variables *interact in a non-additive manner* when affecting a third variable. In other words, the two variables interact to have an effect that is more (or less) than the sum of their parts. 

This might seem pretty abstract, let's use an example to illustrate this.

Let's assume we're trying to predict weight loss of certain people who took a diet, given two (categorical) predictors: the country, and the type of diet they followed. Have a look at the plot below.

- Considering just the people in the USA and the UK, it seems like the effects of both predictors are additive:
    * Weight loss is bigger in the USA than in the UK.
    * Diet C is more effective than diet A. Diet A is more effective than diet B, which makes diet B the least effective.
- When you look at New Zealand, however, it seems like the average weight loss is somewhere between the weight loss for USA and UK, but people seem to be responding much better to diet A in the UK

<img src='./images/new_diet_image.png' width="500">

This means that the "Country" and "Diet" affect weight loss in a non-additive matter. If we're mostly interested in the effect of diet on weight loss (which seems to be plausible here), we say that "Country" is a **confounding factor** of the effect of "Diet" on weight loss.

## Why is it important to account for interactions?

Now that you've seen how interactions work, let's discuss why it is important to add interaction terms. The reason for that is pretty straightforward: not accounting for them might lead to results that are wrong. You'll also notice that including them when they're needed will increase your $R^2$ value!

In our example, the interaction plot was composed out of categorical predictors (countries and diet type), but interactions can occur between categorical variables or between a mix of categorical variables and continuous variables!

Let's go back to our `cars` dataset and look at some interactions we can include: 


```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from itertools import combinations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('auto-mpg.csv') 
```

Going forward, we will be following best practices when preprocessing our data, which includes using scikit-learn transformers. 


```python
# Import transformers
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer

# Instantiate a Min-max scaler
min_max_scaler = MinMaxScaler()

# Instantiate a Standard scaler
standard_scaler = StandardScaler()

# Instantiate a custom transformer for log transformation 
log_transformer = FunctionTransformer(np.log, validate=True)
```


```python
# Columns to be log transformed 
log_columns = ['displacement', 'horsepower', 'weight'] 

# Log transform the columns and convert them into a DataFrame 
log_transformed_df = pd.DataFrame(log_transformer.fit_transform(data[log_columns]), 
                                  columns=['log_disp', 'log_hp', 'log_wt'])
```


```python
# Concat with the original data
log_transformed_df = pd.concat([data, log_transformed_df], axis=1)
log_transformed_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model year</th>
      <th>origin</th>
      <th>car name</th>
      <th>log_disp</th>
      <th>log_hp</th>
      <th>log_wt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
      <td>5.726848</td>
      <td>4.867534</td>
      <td>8.161660</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
      <td>5.857933</td>
      <td>5.105945</td>
      <td>8.214194</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
      <td>5.762051</td>
      <td>5.010635</td>
      <td>8.142063</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
      <td>5.717028</td>
      <td>5.010635</td>
      <td>8.141190</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
      <td>5.710427</td>
      <td>4.941642</td>
      <td>8.145840</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Scale the columns and convert them into a DataFrame
scaled_columns_1 = pd.DataFrame(standard_scaler.fit_transform(log_transformed_df[['log_disp', 'log_wt']]), 
                                columns=['scaled_disp', 'scaled_wt'])

scaled_columns_2 = pd.DataFrame(min_max_scaler.fit_transform(log_transformed_df[['log_hp', 'acceleration']]), 
                                columns=['scaled_hp', 'scaled_acc'])
```


```python
# Concat all the columns into a single DataFrame
preprocessed_df = pd.concat([log_transformed_df[['mpg', 'cylinders', 'model year', 'origin']], 
                             scaled_columns_1, scaled_columns_2], axis=1)
```


```python
# Split into predictor and target variables
y = preprocessed_df[['mpg']]
X = preprocessed_df.drop(['mpg'], axis=1)
```


```python
regression = LinearRegression()
crossvalidation = KFold(n_splits=3, shuffle=True, random_state=1)

baseline = np.mean(cross_val_score(regression, X, y, scoring='r2', cv=crossvalidation))
baseline
```




    0.8322880898272431



See how we built a baseline model using some log-transformed predictors and some categorical predictors. We didn't properly convert the categorical variables to categorical yet, which we should do in the end, but we want to start with a baseline model and a baseline $R^2$ just to get a sense of what a baseline model looks like.

## Interactions between horsepower and origin

To look at how horsepower and origin interact, you can work as follows. Split the data into 3 datasets, one set per origin. Then fit a model with outcome "mpg" and only horsepower as a predictor and do this for each of the dataset. Then plot the data all together and see what the regression lines look like.


```python
origin_1 = preprocessed_df[preprocessed_df['origin'] == 1]
origin_2 = preprocessed_df[preprocessed_df['origin'] == 2]
origin_3 = preprocessed_df[preprocessed_df['origin'] == 3]
origin_1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>model year</th>
      <th>origin</th>
      <th>scaled_disp</th>
      <th>scaled_wt</th>
      <th>scaled_hp</th>
      <th>scaled_acc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>70</td>
      <td>1</td>
      <td>1.125829</td>
      <td>0.720986</td>
      <td>0.645501</td>
      <td>0.238095</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>70</td>
      <td>1</td>
      <td>1.372223</td>
      <td>0.908047</td>
      <td>0.793634</td>
      <td>0.208333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>70</td>
      <td>1</td>
      <td>1.191999</td>
      <td>0.651205</td>
      <td>0.734414</td>
      <td>0.178571</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>70</td>
      <td>1</td>
      <td>1.107370</td>
      <td>0.648095</td>
      <td>0.734414</td>
      <td>0.238095</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>70</td>
      <td>1</td>
      <td>1.094964</td>
      <td>0.664652</td>
      <td>0.691546</td>
      <td>0.148810</td>
    </tr>
  </tbody>
</table>
</div>




```python
regression_1 = LinearRegression()
regression_2 = LinearRegression()
regression_3 = LinearRegression()

horse_1 = origin_1[['scaled_hp']]
horse_2 = origin_2[['scaled_hp']]
horse_3 = origin_3[['scaled_hp']]

regression_1.fit(horse_1, origin_1['mpg'])
regression_2.fit(horse_2, origin_2['mpg'])
regression_3.fit(horse_3, origin_3['mpg'])

# Make predictions
pred_1 = regression_1.predict(horse_1)
pred_2 = regression_2.predict(horse_2)
pred_3 = regression_3.predict(horse_3)

# The coefficients
print(regression_1.coef_)
print(regression_2.coef_)
print(regression_3.coef_)
```

    [-25.29729196]
    [-28.17074549]
    [-30.73684923]


You can see that we have three different estimates for the slope,  -25.29 for origin = 1, -28.17 for origin = 2, and -30.74 for origin = 3. It is not unexpected to see that they are slightly different. Now, let's look at the plot.


```python
# Plot outputs
plt.figure(figsize=(10,6))

plt.scatter(horse_1, origin_1['mpg'],  color='blue', alpha=0.3, label = 'origin = 1')
plt.scatter(horse_2, origin_2['mpg'],  color='red', alpha=0.3, label = 'origin = 2')
plt.scatter(horse_3, origin_3['mpg'],  color='orange', alpha=0.3, label = 'origin = 3')

plt.plot(horse_1, pred_1, color='blue', linewidth=2)
plt.plot(horse_2, pred_2, color='red', linewidth=2)
plt.plot(horse_3, pred_3, color='orange', linewidth=2)
plt.ylabel('mpg')
plt.xlabel('horsepower')
plt.legend();
```


![png](index_files/index_26_0.png)


Even though we get three different lines at different levels, they do seem to be more or less parallel, so the effect seems pretty additive. Just based on looking at this, it seems like there is no real interaction and the effect of origin when predicting mpg using horsepower is additive. It might not be necessary to include an interaction effect in our model. But how would you actually include interaction effects in our model? To do this, you basically multiply 2 predictors. Let's add an interaction effect between origin and horsepower and see how it affects our $R^2$. 


```python
regression = LinearRegression()
crossvalidation = KFold(n_splits=3, shuffle=True, random_state=1)

X_interact = X.copy()
X_interact['hp_origin'] = X['scaled_hp'] * X['origin']

interact_horse_origin = np.mean(cross_val_score(regression, X_interact, y, scoring='r2', cv=crossvalidation))
interact_horse_origin
```




    0.8416810370430339



By actually including an interaction effect here, we did bump our $R^2$ to 0.841 from 0.832, so about 1%! Let's now run the same model in `statsmodels` to see if the interaction effect is significant.


```python
import statsmodels.api as sm
X_interact = sm.add_constant(X_interact)
model = sm.OLS(y, X_interact)
results = model.fit()

results.summary()
```

    //anaconda3/lib/python3.7/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.856</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.853</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   285.6</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 31 Jan 2020</td> <th>  Prob (F-statistic):</th> <td>2.84e-156</td>
</tr>
<tr>
  <th>Time:</th>                 <td>11:51:35</td>     <th>  Log-Likelihood:    </th> <td> -980.75</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   392</td>      <th>  AIC:               </th> <td>   1980.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   383</td>      <th>  BIC:               </th> <td>   2015.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     8</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>       <td>  -31.3053</td> <td>    4.546</td> <td>   -6.886</td> <td> 0.000</td> <td>  -40.244</td> <td>  -22.367</td>
</tr>
<tr>
  <th>cylinders</th>   <td>   -0.0596</td> <td>    0.296</td> <td>   -0.201</td> <td> 0.841</td> <td>   -0.642</td> <td>    0.523</td>
</tr>
<tr>
  <th>model year</th>  <td>    0.7318</td> <td>    0.046</td> <td>   16.002</td> <td> 0.000</td> <td>    0.642</td> <td>    0.822</td>
</tr>
<tr>
  <th>origin</th>      <td>    3.1788</td> <td>    0.563</td> <td>    5.647</td> <td> 0.000</td> <td>    2.072</td> <td>    4.286</td>
</tr>
<tr>
  <th>scaled_disp</th> <td>   -1.0423</td> <td>    0.738</td> <td>   -1.412</td> <td> 0.159</td> <td>   -2.494</td> <td>    0.409</td>
</tr>
<tr>
  <th>scaled_wt</th>   <td>   -3.3236</td> <td>    0.610</td> <td>   -5.452</td> <td> 0.000</td> <td>   -4.522</td> <td>   -2.125</td>
</tr>
<tr>
  <th>scaled_hp</th>   <td>    1.6583</td> <td>    3.342</td> <td>    0.496</td> <td> 0.620</td> <td>   -4.912</td> <td>    8.228</td>
</tr>
<tr>
  <th>scaled_acc</th>  <td>   -3.8734</td> <td>    1.680</td> <td>   -2.306</td> <td> 0.022</td> <td>   -7.177</td> <td>   -0.570</td>
</tr>
<tr>
  <th>hp_origin</th>   <td>   -6.9316</td> <td>    1.446</td> <td>   -4.793</td> <td> 0.000</td> <td>   -9.775</td> <td>   -4.088</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>51.231</td> <th>  Durbin-Watson:     </th> <td>   1.542</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 102.044</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.725</td> <th>  Prob(JB):          </th> <td>6.94e-23</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.036</td> <th>  Cond. No.          </th> <td>2.54e+03</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 2.54e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



Even though the lines look parallel, the interaction effect between horsepower and origin is significant and the R^2 value increased to 0.841.

## Interactions between horsepower and model year

How about interactions between two continuous variables? Let's explore interactions between horsepower and model year. We're interested to see if the effect of horsepower on mpg is different between older and younger cars. 


```python
yr_old = preprocessed_df[:180] # Cars from 70 to 75
yr_young = preprocessed_df[180:] # Cars from 76 to 82
```

What we did here is split the data up to create two datasets, for "older" cars and "younger" cars.


```python
plt.figure(figsize=(12,7))

regression_1 = LinearRegression()
regression_2 = LinearRegression()

horse_1 = yr_old[['scaled_hp']]
horse_2 = yr_young[['scaled_hp']]

regression_1.fit(horse_1, yr_old['mpg'])
regression_2.fit(horse_2, yr_young['mpg'])

# Make predictions
pred_1 = regression_1.predict(horse_1)
pred_2 = regression_2.predict(horse_2)

# The coefficients
print(regression_1.coef_)
print(regression_2.coef_)
```

    [-21.39522143]
    [-35.10169206]



    <Figure size 864x504 with 0 Axes>



```python
# Plot outputs
plt.figure(figsize=(10,6))

plt.scatter(horse_1, yr_old['mpg'], color='blue', alpha=0.3, label='older cars')
plt.scatter(horse_2, yr_young['mpg'], color='red', alpha=0.3, label='younger cars')

plt.plot(horse_1, pred_1, color='blue', linewidth=2)
plt.plot(horse_2, pred_2, color='red', linewidth=2)

plt.ylabel('mpg')
plt.xlabel('horsepower')
plt.legend();
```


![png](index_files/index_37_0.png)


More than for our previous example. there seems to be an interaction between horsepower and cars. Let's add the interaction effect in our model and see how it affects $R^2$. 


```python
regression = LinearRegression()
crossvalidation = KFold(n_splits=3, shuffle=True, random_state=1)

X_interact_2 = X.copy()
X_interact_2['horse_year'] = X['scaled_hp'] * X['model year']

interact_horse_origin = np.mean(cross_val_score(regression, X_interact_2, y, scoring='r2', cv=crossvalidation))
interact_horse_origin
```




    0.8597777940161037



This result confirms what we have seen before: including this interaction has an even bigger effect on the $R^2$. When running this in `statsmodels`, unsurprisingly, the effect is significant.


```python
import statsmodels.api as sm
X_interact_2 = sm.add_constant(X_interact_2)
model = sm.OLS(y,X_interact_2)
results = model.fit()

results.summary()
```

    //anaconda3/lib/python3.7/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.874</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.871</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   331.7</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 31 Jan 2020</td> <th>  Prob (F-statistic):</th> <td>5.05e-167</td>
</tr>
<tr>
  <th>Time:</th>                 <td>11:51:35</td>     <th>  Log-Likelihood:    </th> <td> -955.36</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   392</td>      <th>  AIC:               </th> <td>   1929.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   383</td>      <th>  BIC:               </th> <td>   1964.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     8</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>       <td>  -87.1803</td> <td>    7.725</td> <td>  -11.286</td> <td> 0.000</td> <td> -102.368</td> <td>  -71.993</td>
</tr>
<tr>
  <th>cylinders</th>   <td>    0.3899</td> <td>    0.260</td> <td>    1.498</td> <td> 0.135</td> <td>   -0.122</td> <td>    0.902</td>
</tr>
<tr>
  <th>model year</th>  <td>    1.4885</td> <td>    0.095</td> <td>   15.590</td> <td> 0.000</td> <td>    1.301</td> <td>    1.676</td>
</tr>
<tr>
  <th>origin</th>      <td>    0.6130</td> <td>    0.256</td> <td>    2.396</td> <td> 0.017</td> <td>    0.110</td> <td>    1.116</td>
</tr>
<tr>
  <th>scaled_disp</th> <td>   -0.9066</td> <td>    0.689</td> <td>   -1.315</td> <td> 0.189</td> <td>   -2.262</td> <td>    0.449</td>
</tr>
<tr>
  <th>scaled_wt</th>   <td>   -3.7671</td> <td>    0.561</td> <td>   -6.712</td> <td> 0.000</td> <td>   -4.871</td> <td>   -2.664</td>
</tr>
<tr>
  <th>scaled_hp</th>   <td>  120.4052</td> <td>   14.754</td> <td>    8.161</td> <td> 0.000</td> <td>   91.395</td> <td>  149.415</td>
</tr>
<tr>
  <th>scaled_acc</th>  <td>   -2.2225</td> <td>    1.575</td> <td>   -1.411</td> <td> 0.159</td> <td>   -5.319</td> <td>    0.874</td>
</tr>
<tr>
  <th>horse_year</th>  <td>   -1.7272</td> <td>    0.194</td> <td>   -8.896</td> <td> 0.000</td> <td>   -2.109</td> <td>   -1.345</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>32.600</td> <th>  Durbin-Watson:     </th> <td>   1.580</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  67.171</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.465</td> <th>  Prob(JB):          </th> <td>2.59e-15</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.802</td> <th>  Cond. No.          </th> <td>9.72e+03</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 9.72e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



## Additional resources

- You can use the Python library `seaborn` to visualize interactions as well. Have a look [here](https://blog.insightdatascience.com/data-visualization-in-python-advanced-functionality-in-seaborn-20d217f1a9a6) for more information.

- [This resource](http://www.medicine.mcgill.ca/epidemiology/joseph/courses/EPIB-621/interaction.pdf) walks over multiple examples of regressions with interaction terms. Even though the code is in R, it might give you some additional insights into how interactions work.


## Summary

Great! You now know how to interpret interactions, how to include them in your model and how to interpret them. Obviously, nothing stops you from adding multiple interactions at the same time, and you probably should for many occasions. You'll practice what you learned here in the next lab, including interactions.  
