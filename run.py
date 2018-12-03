import sys
import getopt
import getpass
import pandas as pd
from algorithm.regression import SKLearnAlgorithm
from algorithm.mimic_sklearn import MimicSKLearn
from algorithm.sperate_model import SeperateModel

training_data = pd.read_csv("data/train.csv")
testing_data = pd.read_csv("data/test.csv")
new_less_training_data = pd.read_csv("data/train_with_my_time_less.csv")
new_less_testing_data = pd.read_csv("data/test_with_my_time_less.csv")

def main(argv):
    with SKLearnAlgorithm(training_data=new_less_training_data, testing_data=new_less_testing_data) as algorithm:
        #algorithm.run_with_validation()
        algorithm.run_without_validation()
        #algorithm.build_model_for_each_penalty()
        #algorithm.feature_ranking()

    # with MimicSKLearn(training_data=training_data, testing_data=testing_data) as algorithm:
    #     algorithm.mimic_process()
    #     algorithm.mimic_test_process()

    #with SeperateModel(training_data=new_training_data, testing_data=testing_data) as algorithm:
        #algorithm.build_model_for_each_penalty()
        #algorithm.build_model_for_each_penalty_skera()
        

if __name__ == '__main__':
    main(sys.argv[1:])
