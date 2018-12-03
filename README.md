##  Kaggle competition about https://www.kaggle.com/c/msbd5001-fall2018/data

### Data

Data is located in `Data` folder. 

#### File descriptions
- Data/train.csv - the training set
- Data/test.csv - the test set
- submission/submission.csv - submission file in the correct format

#### Data fields.
`'Time'` is the training time of the model. The other feature follows the definition in Sklearn. Specifically, `'n_samples','n_features','n_classes','n_clusters_per_class','n_informative','flip_y','scale'` describes how the synthetic dataset for training is generated using [sklearn.datasets.make_classification](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html), and `'penalty','l1_ratio','alpha','max_iter','random_state','n_jobs'` describes the setup of [sklearn.linear_model.SGDClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier).

### Programming languages
Python 3.6

### Enviroment Setup via virtualenv

- The project is run in Mac and needs pip, virtualenv and python 3.6.

- Apply virtualenv. (https://docs.python-guide.org/dev/virtualenvs/)
```zsh
$ cd my_project_folder
$  my_project
```

- Use virtualenv for Python Development, Need Python 3.6.
```zsh
# use the Python interpreter of your choice (like python3.6).
# check swhere is your python 3
$ which python3
$ virtualenv -p /anaconda3/bin/python IndividualProject
```

- Activate virtual environment
```zsh
$ source my_project/bin/activate
```

- Deactivate:
```zsh
$ deactivate
```

## Running the Project
 1. Activate the virtual environment.

 2. If it is your first time running the client, install the Python dependencies:
```zsh
 chmod u+x setup
 ./setup
```
 from the root of the project. It will read all dependencies from `requirements.txt` file and run pip to install them.

 3. Run the process:
 ```zsh
 chmod u+x run
 ./run
```

It will generated a "result.cvs" file which contains the prediction. 

**Please note that you will need to open the file to add `Id` as the first column name!! Otherwise Kaggle will not accept it**

## Project Introduction

### Files Structure

- **/.vscode**
Contain the setting for VS Code.

- **/algorithm**
    1. **regression.py**: It contains the final model that creates the result which I submitted to Kaggle in the end. It used Nerual Network to train the data via L1 Regularization.
    2. **mimic_sklearn.py**: It contains the script to generated referenced time value to improve the training.
    3. **sperate_model.py**: It's the model that train nerual network for each kind of penaly. But it's not used in the final submission.

- **/Data**
Contain the Training data set and Testing data set.

- **run.py**
This is the main entry of the project. We can change it to run different models and generate the result.

### Preprocessing

1. Generate fake time data to help model find the rules. 
 - Run `MimicSKLearn` model in Line 21 to 23 of `run.py` for training data and testing data. Get local time feature.
 - Add local time feature into the training data set and testing data set.

2. Do feature selection to help remove the outliers.
 - Run `SKLearnAlgorithm -> algorithm.feature_ranking()` in Line 19 of `run.py` to get the feature ranking.
 - Remove less important features in training data set and testing data set.

3. After above 2 steps, we get new training data set **(i.e. data/train_with_my_time_less.csv)** and new testing data set **(i.e. data/test_with_my_time_less.csv)**

4. Run `SKLearnAlgorithm -> algorithm.run_without_validation()` in Line 17 of `run.py` to get few different results.

5. Do average for each records then come up with the final `/submission/submission.csv`. 