# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Deepthi B

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
Experiments performed were:
1. Initial Raw Submission   **[Model: `initial`]**
2. Added Features Submission *(EDA +  Feature Engineering)* **[Model: `add_features`]**
3. Hyperparameter Optimization (HPO) - Initial Setting Submission 
4. Hyperparameter Optimization (HPO) - Setting 1 Submission 
5. Hyperparameter Optimization (HPO) - Setting 2 Submission **[Model: `hpo (top-hpo-model: hpo2)`]**

My Observations:
Some experiments provided negative predictions values.

Changes made:
All negatives were replaced by 0.
### What was the top ranked model that performed?
The top-performing model, named WeightedEnsemble_L3, emerged as the standout among the models developed, boasting a validation RMSE score of 34.0449 and achieving the best Kaggle score of 0.44716 on the test dataset. Notably, this model was crafted through training on data enriched with exploratory data analysis (EDA) and feature engineering, without resorting to hyperparameter optimization. Despite the potential for improved RMSE scores on the validation data observed with some models after hyperparameter optimization, it was this particular model that excelled on the unseen test dataset. The decision to prioritize this model was based on its impressive performance in both RMSE (cross-validation) and Kaggle (test data) scores.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
- The 'datetime' feature was processed as a datetime object to extract hour information from the timestamp.
- Initially encoded as integers, the independent features 'season' and 'weather' were converted to the categorical data type to reflect their categorical nature.
- Using feature extraction, independent features such as 'year', 'month', 'day' (dayofweek), and 'hour' were derived from the 'datetime' feature. Subsequently, the 'datetime' feature was removed from the dataset.
- Upon examination, it was found that the inclusion of the features 'casual' and 'registered' significantly improved RMSE scores during cross-validation, indicating a high correlation with the target variable 'count'. However, since these features were only present in the training dataset and absent in the test data, they were disregarded during model training.
- A new categorical feature, 'day_type', was created based on the existing features 'holiday' and 'workingday', effectively categorizing days into "weekday", "weekend", and "holiday".
- Due to their high positive correlation of 0.98, the features 'temp' (temperature in degrees Celsius) and 'atemp' (feels like temperature in degrees Celsius) were found to be highly correlated. To alleviate multicollinearity between independent variables, 'atemp' was dropped from both the train and test datasets.
- Data visualization techniques were employed to gain additional insights into the dataset.

### How much better did your model preform after adding additional features and why do you think that is?
- The inclusion of additional features resulted in a substantial enhancement of model performance, exceeding 130% improvement compared to the performance of the initial/raw model (prior to any exploratory data analysis or feature engineering).
- Model performance saw an improvement when certain categorical variables, initially encoded as integers, were converted to their proper categorical data types.
- Furthermore, during model training, the features 'casual' and 'registered' were disregarded, and 'atemp' was removed from the datasets due to its high correlation with another independent variable, 'temp'. This action aided in mitigating multicollinearity.
- Additionally, breaking down the 'datetime' feature into multiple independent features such as 'year', 'month', 'day', and 'hour', alongside the introduction of 'day_type', contributed to improved model performance. These predictor variables facilitate the model in assessing seasonality or historical patterns within the data more accurately.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
Hyperparameter tuning proved advantageous by boosting the model's performance compared to its initial submission. Three distinct configurations were tested during the optimization experiments. Despite the competitive performance of the tuned models, when compared to the model incorporating exploratory data analysis (EDA) and additional features, the latter exhibited significantly superior performance on the Kaggle test dataset.

Incorporating the prescribed settings, the autogluon package was employed for training. However, the performance of hyperparameter-optimized models fell short of expectations due to the fixed set of values used for tuning, restricting the range of options autogluon could explore.

During hyperparameter optimization with autogluon, parameters like 'time_limit' and 'presets' played pivotal roles. Autogluon might fail to construct any models within the specified hyperparameter set if the time limit is insufficient for model construction.

Additionally, hyperparameter optimization with presets such as "high_quality" (with auto_stack enabled) demands high memory usage and computational intensity within the given time limit and available resources. Consequently, lighter and faster preset options like 'medium_quality' and 'optimized_for_deployment' were tested. I opted for the faster and lighter preset, "optimized_for_deployment," for the hyperparameter optimization routine, as the others failed to generate models using AutoGluon for the experimental configurations.

The balance between exploration and exploitation poses the greatest challenge when utilizing AutoGluon with a prescribed range of hyperparameters.
### If you were given more time with this dataset, where do you think you would spend more time?
With more time, I want to investigate additional outcomes by running Autogluon with high quality preset and enhanced hyperparameter tuning.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|prescribed_values|prescribed_values|"presets: 'high quality' (auto_stack=True)"|1.79874|
|add_features|prescribed_values|prescribed_values|"presets: 'high quality' (auto_stack=True)"|0.44716|
|hpo (top-hpo-model: hpo2)|Tree-Based Models: (GBM, XT, XGB & RF)|KNN|"presets: 'optimize_for_deployment"|0.49927|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

TODO: Replace the image below with your own.

![model_train_score.png](https://drive.google.com/file/d/1jRjdSrn7o8FQGsnHwTSUu8f74OLYawvw/view?usp=sharing)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![model_test_score.png](https://drive.google.com/file/d/1O-kFAbWQ1ToikJlqYunrC3G-pLfntEQN/view?usp=sharing)

## Summary
-Thorough examination and integration of the AutoGluon AutoML framework for Tabular Data were conducted within the bike sharing demand prediction project.
-The full spectrum of capabilities within the AutoGluon framework was harnessed to construct automated stack ensembles and individually tailored regression models trained specifically on tabular data. This facilitated the rapid prototyping of a baseline model.
-The highest-performing model, based on AutoGluon, significantly enhanced results by incorporating insights from extensive exploratory data analysis (EDA) and feature engineering, without the need for hyperparameter optimization.
-AutoGluon's utilization of automatic hyperparameter tuning, model selection/ensembling, and architecture search enabled exhaustive exploration and exploitation of the most promising options.
-Moreover, hyperparameter tuning with AutoGluon demonstrated improved performance over the initial raw submission. However, it fell short of the model incorporating EDA, feature engineering, and lacking hyperparameter tuning.
-Notably, the process of hyperparameter tuning using AutoGluon, without default hyperparameters or random parameter configurations, was observed to be intricate. Its efficacy depended heavily on factors such as time constraints, prescribed presets, the potential model family, and the range of hyperparameters under consideration.
       
     

