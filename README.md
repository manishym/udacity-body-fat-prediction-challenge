udacity-body-fat-prediction-challenge
==============================
## Intro
This is the udacity bodyfat challenge. Most off the work is in 'udacity-body-fat-prediction-challenge.ipynb'.
Our approach was to augment the data and then use a bunch of models to stack predictions made on a kfold fold.

### Data Augmentation
To augment the data, we have used two methods.
1. Feature augmentation
    Basically, body fat is more dependent on the ratios of measurements as opposed to actual measurement. For example, if person A has 40 inch chest and 38 inch waist and person B has 44 inch chest and 38 inch waist, we can reasonably assume that person A has more body fat compared to person B. Similar analogy for height to weight ratio, bicep to forearm ratio, hip to waist ratio etc. Initially we selected manual ratios, then we decided to take ratio of all measurements wrt each other.
2. Row augmentation
    The train data had just 200 rows. We made 4 folds in kfold. Hence we just had 150 rows for training. To augment this, we multiplied each of the values with a random number from -1.05 to + 1.05 and added the rows back. This improved validation score, however after private leaderboard was announced, we realized it did not help.
3. Outliers
    There were few outilers and we used zscore to remove the outliers.


### Cross Validation
We created cross validation set using KFold cross validation from sklearn and measured the models using r2_score and mae score.

### Models
For one of the submissions we used
RandomForestRegressor, GradientBoostingRegressor, LGBMRegressor, XGBRegressor, AdaBoostRegressor, Ridge and Lasso.
In another submission, we used all the above models and a neural network using pytorch.
The network had an input layer of 128 nodes, hidden layers of 64 and 32 nodes and an output layer of one node.


### Hyperparameter Tuning
We used sklearn GridSearchCV to individually find the best hyperparameters for all the models. Then we used the best model for each of them. We used sklearn's clone function to create model with hyperparameter but not weights in each iteration of kfold split.



Project Organization
------------
 
    ├── LICENSE
    ├── Makefile           <- Makefile with commands that perform parts of the processing pipeline
    ├── README.md          <- The top-level README
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── Dockerfile         <- Dockerfile, alternative approach to manage environment
    │                         more interesting if using non-Unix
    ├── submissions        <- Directory to keep submissions
    │ 
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions for submissions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
