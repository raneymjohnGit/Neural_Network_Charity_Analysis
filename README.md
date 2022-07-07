# Neural_Network_Charity_Analysis
To Complete Neural Networks Machine Learning Models from Module 19

## Project Overview
With the knowledge of machine learning and neural networks, we have to use the features in the provided dataset to help create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.From Alphabet Soup’s business team, we received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

-   EIN and NAME—Identification columns
-   APPLICATION_TYPE—Alphabet Soup application type
-   AFFILIATION—Affiliated sector of industry
-   CLASSIFICATION—Government organization classification
-   USE_CASE—Use case for funding
-   ORGANIZATION—Organization type
-   STATUS—Active status
-   INCOME_AMT—Income classification
-   SPECIAL_CONSIDERATIONS—Special consideration for application
-   ASK_AMT—Funding amount requested
-   IS_SUCCESSFUL—Was the money used effectively

## Resources
- jupyter notebook, python, Machine Learning Models 

## Challenge Overview
Prerequisite:
1.  Download the AlphabetSoupCharity_starter_code.ipynb
2.  Download the data in csv charity_data.csv for this excercise 


## Deliverable 1:  Preprocessing Data for a Neural Network Model

Follow the instructions below and use the AlphabetSoupCharity_starter_code.ipynb file to complete Deliverable 1.

1.  Open the AlphabetSoupCharity_starter_code.ipynb file, rename it AlphabetSoupCharity.ipynb, and save it to your Neural_Network_Charity_Analysis GitHub folder.
2.  Using the information we have provided in the starter code, follow the instructions to complete the preprocessing steps.
3.  Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
	a.  What variable(s) are considered the target(s) for your model?
	b.  What variable(s) are considered the feature(s) for your model?
4.  Drop the EIN and NAME columns.
5.  Determine the number of unique values for each column.
6.  For those columns that have more than 10 unique values, determine the number of data points for each unique value.
7.  Create a density plot to determine the distribution of the column values.
8.  Use the density plot to create a cutoff point to bin "rare" categorical variables together in a new column, Other, and then check if the binning was successful.
9.  Generate a list of categorical variables.
10. Encode categorical variables using one-hot encoding, and place the variables in a new DataFrame.
11. Merge the one-hot encoding DataFrame with the original DataFrame, and drop the originals.
12. Split the preprocessed data into features and target arrays.
13. Split the preprocessed data into training and testing datasets.
14. Standardize numerical variables using Scikit-Learn’s StandardScaler class, then scale the data.
15. Save your AlphabetSoupCharity.ipynb file to your Neural_Network_Charity_Analysis folder.

## Deliverable 2:  Compile, Train, and Evaluate the Model

Follow the instructions below and use the information file to complete Deliverable 2.

1.  Continue using the AlphabetSoupCharity.ipynb file where you’ve already performed the preprocessing steps from Deliverable 1.
2.  Create a neural network model by assigning the number of input features and nodes for each layer using Tensorflow Keras.
3.  Create the first hidden layer and choose an appropriate activation function.
4.  If necessary, add a second hidden layer with an appropriate activation function.
5.  Create an output layer with an appropriate activation function.
6.  Check the structure of the model.
7.  Compile and train the model.
8.  Create a callback that saves the model's weights every 5 epochs.
9.  Evaluate the model using the test data to determine the loss and accuracy.
10. Save and export your results to an HDF5 file, and name it AlphabetSoupCharity.h5.

## Deliverable 3:  Optimize the Model

Using your knowledge of TensorFlow, optimize your model in order to achieve a target predictive accuracy higher than 75%. 
If you can't achieve an accuracy higher than 75%, you'll need to make at least three attempts to do so.

1.  Optimize your model in order to achieve a target predictive accuracy higher than 75% by using any or all of the following:
    -   Adjusting the input data to ensure that there are no variables or outliers that are causing confusion in the model, such as:
    -   Dropping more or fewer columns.
    -   Creating more bins for rare occurrences in columns.
    -   Increasing or decreasing the number of values for each bin.
    -   Adding more neurons to a hidden layer.
    -   Adding more hidden layers.
    -   Using different activation functions for the hidden layers.
    -   Adding or reducing the number of epochs to the training regimen.

2.  Create a new Jupyter Notebook file and name it AlphabetSoupCharity_Optimzation.ipynb.
3.  Import your dependencies, and read in the charity_data.csv to a Pandas DataFrame.
4.  Preprocess the dataset like you did in Deliverable 1, taking into account any modifications to optimize the model.
5.  Design a neural network model, taking into account any modifications that will optimize the model to achieve higher than 75% accuracy.
6.  Create a callback that saves the model's weights every 5 epochs.
7.  Save and export your results to an HDF5 file, and name it AlphabetSoupCharity_Optimization.h5.


## Neural_Network_Charity_Analysis Results
1.  Data Preprocessing
    -   What variable(s) are considered the target(s) for your model?
        IS_SUCCESSFUL is the bvraible used as target for the models
    
    -   What variable(s) are considered to be the features for your model?
        The following are considered features for the model

        *   APPLICATION_TYPE—Alphabet Soup application type
        *   AFFILIATION—Affiliated sector of industry
        *   CLASSIFICATION—Government organization classification
        *   USE_CASE—Use case for funding
        *   ORGANIZATION—Organization type
        *   STATUS—Active status
        *   INCOME_AMT—Income classification
        *   SPECIAL_CONSIDERATIONS—Special consideration for application
        *   ASK_AMT—Funding amount requested        

    -   What variable(s) are neither targets nor features, and should be removed from the input data?

        *   EIN and NAME—Identification columns

2.  Compiling, Training, and Evaluating the Model
    -   How many neurons, layers, and activation functions did you select for your neural network model, and why?
        After several trial and errors , we came up wih that number of neurons as 80/40/1, layers as 3 and activation function relu and sigmoid.
    
    -   Were you able to achieve the target model performance?    
        No. We tried several combinations ans could not achieve 75%
    
    -   What steps did you take to try and increase model performance?
        -   We tried removeing no impact columns such as INCOME_AMT
        -   Reduced the number of epochs


## Neural_Network_Charity_Analysis Summary

1.  Deep Learning model may not be right model for this data set as the accuracy is less than 75%
2.  Ensemble alogorithm "Easy Ensemble AdaBoost Classifier" may be more suitable as it has more accuracy in our previous runs.
