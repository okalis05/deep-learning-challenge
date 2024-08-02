# deep-learning-challenge
---

## Credit Risk Classification report
 ---

### Overview of the Analysis
---
The nonprofit foundation `Alphabet Soup` wants a tool that can help select applicants for funding with the best chances of success in  their ventures and we were tasked to use the features in the provided dataset to create a binary classifier that can predict whether or not applicants will be successful if funded by `Alphabet Soup`.

---
### Purpose of the analysis
---
The purpose of this analysis wass to build a deep learning model able to predict the outcome of business ventures funded by `Alphabet Soup` with a  targetted predictive accuracy score of 75% or higher.

### Methodology
---
In order to build our model , the following steps were taken.
* Pre_processing the Data using  Pandas and scikit-learn.
* Compiling, Training, and Evaluating the Model using TensorFlow.
* Optimizing the Model using TensorFlow.
* Repeating the process and adjusting the parameters until the targeted accuracy score is reached
  
---
### Results
---

#### 1-Preprocessing
---
- First we imported the necessary dependencies , read and stored the `charity_data.csv ` file into a pandas dataframe named `application_df`.
- Then dropped the columns 'EIN' and 'NAME' which weren't going to serve as target and weren't contributing to the analysis as features.
- Set the `IS_SUCCESSFUL` column as our target column and the remaining columns , as `Features` used to determine classify our target.
- We also determined the count of unique values in the features columns , identified the less recurring values in columns'APPLICATION_TYPE' and 'CLASSIFICATION' which we binned into a new category named 'Other'. 
- After that we converted categorical data to numeric with `pd.get_dummies` function.
- Furthermore we splitted our preprocessed data into our features and target arrays then into training and testing datasets.
- Finally we created a `StandardScaler` instance , fitted then scaled the data


#### 2-Compile, Train and Evaluate the Model
---
* In this portion , we completed the following steps.
- Defining the model `nm` with 2 hidden layers and an ouput layer.The first hidden layer was assigned 8 nodes and the activation function `relu`
while the second hidden layer was assigned 5 nodes  and the `sigmoid` activation function and for the output layer we used the `sigmoid` activation
function. All those parameters were chosen randomly as a starting point.
- Compiling and training our model using the X_train_scaled data and epochs = 300 , again chosen randomly as a starting point.
- Evaluating the model using the X_test_scaled data.
* This resulted to a predictive accuracy score of `72.5%` which is lower than the targetted score of at least `75%` , prompting the model optimization.

#### 3-Optimize the Model
---
* In order to upscale our accuracy score , we decided to use the` keras_tuner` library to create a new Sequential model with hyperparameter options.
- First we defined a function  named `create_model(hp)` to return our model, then set it up to pick the best activation functions , the number of layers , and the number of neurons in each layer .Then we compiled the model , imported the `keras_tuner` and ran it , got the best hyperparameters and finally evaluated it.This got us a model with an accuracy score of `72.6%`, almost exactly as our initial model.This finding lead us to the understanding that with our current prepocessing we wouldn't be able to achieve our targetted score ,therefore we had to rethink our initial process.
- With that insight ,  we decided to reduce the bins on the `APPLICATION_TYPE` and `CLASSIFICATION` by setting the cutoff_value to `1065` and `777` respectively. We also increased our training data ratio from 75% to 85% and increased the number of hidden layers from 2 to 4. We assigned 20 nodes to each one of the 4 hidden layers .We also used `relu`  activation function on the input layer , `tanh` on the 3 hidden layers and the `sigmoid` function on the output layer.Finally we increased the epochs number to '500',compiled, trained and evaluated the model .Those new parameters lead to an accuracy score `72.6%` ,which was not any better than the previous optimization attempt.
- As a last resort , we decided to keep the 'EIN' column and almost the same parameters as previously set but instead of `tanh` , we used the `sigmoid` function on the hidden layers , increased the epochs to 800 and increased the training data ratio to 90%.This boosted our model accuracy score to `73.1%`
---
### Summary
---
In summary , we were tasked with creating a Deep Learning Model than can predict the outcome of bussiness ventures funded by the foundation `Alphabet Soup` with and accuracy score of at least `75%`.For that we preprocessed our data , created a deep learning model , compiled ,trained and  evaluated it and made 3 attempts to optimize it in hope of reaching our targetted accuracy score but ultimately we could not get there.Given that this was a binary classification , a logistic regression could have done a better job at predicting the outcome with the expected level of accuracy .
