from __future__ import division
import os,sys
import traceback
import numpy as np
from random import seed, shuffle

import utilss as ut
import pandas as pd
import math

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from pgmpy.models.BayesianModel import BayesianModel
from pgmpy.models.BayesianNetwork import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import TreeSearch, BayesianEstimator
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score
#from pgmpy.factors.continuous import ContinuousNode
#from pgmpy.factors import BaseDiscretizer
from pgmpy.models import NaiveBayes

from tqdm import tqdm
from functools import partialmethod
# verbose = false for tqdm in pgmpy
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)


def train_model_disp_mist(x, y, x_control,NB,index):
    # Creating and training the Complement Naive Bayes Classifier
    if (NB==1):
        classifier = MultinomialNB(alpha=3.0)#GaussianNB()#ComplementNB()
        classifier.fit(x, y)
    elif (NB==2):
        df_train = pd.DataFrame(x)  # df.columns)
        df_train['Y'] = y
        df_train.rename(lambda x: 'col_' + str(x)[0:], axis='columns', inplace=True)

        classifier =NaiveBayes()
        classifier.fit(df_train, 'col_Y')
    elif (NB>=3):
        # learn the TAN graph structure from data
        df_train = pd.DataFrame(x)  # df.columns)
        df_train['Y'] = y
        df_train.rename(lambda x: 'col_' + str(x)[0:], axis='columns', inplace=True)
        #df_train.to_csv('df_train.csv')



        # Structure learning
        #node = df_train.columns[0]
        #model = bn.structure_learning.fit(df_train)
        #model = bn.independence_test(model, df_train)
        # Compute edge strength using chi-square independence test
        #classifier = bn.independence_test(model, df_train, alpha=0.05, prune=True)
        #scoring_method = K2Score(data=df_train)
        #est = HillClimbSearch(data=df_train)
        #dag = est.estimate(scoring_method=scoring_method, max_indegree=4, max_iter=int(1e2) )

        est = TreeSearch(df_train, root_node='col_'+str(index))
        dag = est.estimate(estimator_type='tan', class_node='col_Y')
        # construct Bayesian network by parameterizing the graph structure
        classifier = BayesianModel(dag.edges())
        classifier.fit(df_train, estimator=BayesianEstimator, #prior_type='K2', )
                       prior_type="BDeu",
                       equivalent_sample_size=10,
                       complete_samples_only=False)


        '''
        BNTest = skbn.BNClassifier(prior='Smoothing' ,priorWeight=2)
        #BNTest = skbn.BNClassifier  ( learningMethod='TAN', prior= 'Smoothing', priorWeight = 2, scoringType='BDeu', discretizationStrategy = 'quantile', discretizationNbBins='elbowMethod', discretizationThreshold=25,  usePR = True, significant_digit = 7)

       
        xTrain, yTrain = BNTest.XYfromCSV(filename='df_train.csv', target='col_Y')
        BNTest.fit(xTrain, yTrain)

        classifier = skbn.BNClassifier(significant_digit=7)
        classifier.fromTrainedModel(bn=BNTest.bn, targetAttribute='col_Y', targetModality='1', threshold=BNTest.threshold, variableList=df_train.columns.tolist())
        '''
    return classifier


def get_clf_stats(w, x_train, y_train, x_control_train, x_test, y_test, x_control_test, sensitive_attrs,NB):


    assert(len(sensitive_attrs) == 1) # ensure that we have just one sensitive attribute
    s_attr = list(sensitive_attrs)[0] # for now, lets compute the accuracy for just one sensitive attr

    # Evaluating the classifier
    if(NB>=2):
        df_test = pd.DataFrame(x_test)  # , columns=cols_features)#df.columns)
        #df_test['Y'] = y_test
        df_test.rename(lambda x: 'col_' + str(x), axis='columns', inplace=True)
        #df_test.to_csv('df_test.csv')

        #scoreCSV2 = w.predict(df_test)# , df_test['col_Y'])
        #prediction_test = scoreCSV2.astype(int)
        prediction_test =w.predict(df_test)
        prediction_test=prediction_test['col_Y']

        df_train = pd.DataFrame(x_train)  # , columns=cols_features)#df.columns)
        #df_train['Y'] = y_train
        df_train.rename(lambda x: 'col_' + str(x), axis='columns', inplace=True)
        #df_train.to_csv('df_train.csv')
        #scoreCSV2 = w.predict(df_train )#,  df_train['col_Y'])
        #prediction_train = scoreCSV2.astype(int)
        prediction_train = w.predict(df_train)  #
        prediction_train=prediction_train['col_Y']

       # x_test1 = pd.DataFrame(x_test)  # df.columns)
       # x_test1.rename(lambda x: 'col_' + str(x)[0:], axis='columns', inplace=True)
       # x_train1 = pd.DataFrame(x_train)  # df.columns)
       # x_train1.rename(lambda x: 'col_' + str(x)[0:], axis='columns', inplace=True)
       # prediction_test = np.array(w.predict(x_test1)).reshape(-1)
       # prediction_train = np.array(w.predict(x_train1)).reshape(-1)
    else:
        prediction_test = w.predict(x_test)
        prediction_train = w.predict(x_train)

    # compute distance from boundary
#    distances_boundary_train = get_distance_boundary(w, x_train, x_control_train[s_attr])
#    distances_boundary_test = get_distance_boundary(w, x_test, x_control_test[s_attr])

    # compute the class labels
    all_class_labels_assigned_train = prediction_train# np.sign(distances_boundary_train)
    all_class_labels_assigned_test = prediction_test#np.sign(distances_boundary_test)


    #train_score, test_score, correct_answers_train, correct_answers_test = ut.check_accuracy(None, x_train, y_train, x_test, y_test, all_class_labels_assigned_train, all_class_labels_assigned_test)


    cov_all_train = {}
    cov_all_test = {}
    for s_attr in sensitive_attrs:

        print_stats = False

        # we arent printing the stats for the train set to avoid clutter

        # uncomment these lines to print stats for the train fold
        # print "*** Train ***"
        # print "Accuracy: %0.3f" % (train_score)
        # print_stats = True
        s_attr_to_fp_fn_train = get_fpr_fnr_sensitive_features(y_train, all_class_labels_assigned_train, x_control_train, sensitive_attrs, print_stats)



        #print_stats = True # only print stats for the test fold
        if NB<4:#fine tuning stage
            print_stats = True
        s_attr_to_fp_fn_test = get_fpr_fnr_sensitive_features(y_test, all_class_labels_assigned_test, x_control_test, sensitive_attrs, print_stats)
        #print ("\n")

    return all_class_labels_assigned_train, all_class_labels_assigned_test,  s_attr_to_fp_fn_train, s_attr_to_fp_fn_test



def get_fpr_fnr_sensitive_features(y_true, y_pred, x_control, sensitive_attrs, verbose):


    # we will make some changes to x_control in this function, so make a copy in order to preserve the origianl referenced object
    #x_control_internal = deepcopy(x_control)

    s_attr_to_fp_fn = {}
    
    for s in sensitive_attrs:
        s_attr_to_fp_fn[s] = {}
        s_attr_vals = x_control[s]
        #if verbose == False:#True:
        #    print ("_ |  s  | Acc. | FPR. | FNR. | Rec. | Prc. | F1s. |")
        for s_val in sorted(list(set(s_attr_vals))):
            s_attr_to_fp_fn[s][s_val] = {}
            y_true_local = y_true[s_attr_vals==s_val]
            y_pred_local = y_pred[s_attr_vals==s_val]
            y_pred_local[y_pred_local == 0] = -1
            #print(y_pred_local)

            

            acc = float(np.sum(y_true_local==y_pred_local)) / len(y_true_local)

            fp = np.sum(np.logical_and(y_true_local == -1.0, y_pred_local == +1.0)) # something which is -ve but is misclassified as +ve
            fn = np.sum(np.logical_and(y_true_local == +1.0, y_pred_local == -1.0)) # something which is +ve but is misclassified as -ve
            tp = np.sum(np.logical_and(y_true_local == +1.0, y_pred_local == +1.0)) # something which is +ve AND is correctly classified as +ve
            tn = np.sum(np.logical_and(y_true_local == -1.0, y_pred_local == -1.0)) # something which is -ve AND is correctly classified as -ve

            try:
                recall=tp/(tp+fn)
            except:
                recall=0
            try:
                precision=tp/(tp+fp)
            except:
                precision = 0
            try:
                fscore=2/(1/precision+1/recall)
            except:
                fscore=0


            try:
                fpr = float(fp) / float(fp + tn)
            except:
                fpr=0
            try:
                fnr = float(fn) / float(fn + tp)
            except:
                fnr=0
            try:
                tpr = float(tp) / float(tp + fn)
            except:
                tpr=0
            try:
                tnr = float(tn) / float(tn + fp)
            except:
                tnr=0

            mcc = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            s_attr_to_fp_fn[s][s_val]["mcc"] = mcc

            L = len(y_pred_local)
            s_attr_to_fp_fn[s][s_val]["L"] = L

            s_attr_to_fp_fn[s][s_val]["fp"] = fp/L
            s_attr_to_fp_fn[s][s_val]["fn"] = fn/L
            s_attr_to_fp_fn[s][s_val]["tp"] = tp/L
            s_attr_to_fp_fn[s][s_val]["tn"] = tn/L

            s_attr_to_fp_fn[s][s_val]["fpr"] = fpr
            s_attr_to_fp_fn[s][s_val]["fnr"] = fnr
            s_attr_to_fp_fn[s][s_val]["tpr"] = tpr
            s_attr_to_fp_fn[s][s_val]["tnr"] = tnr

            s_attr_to_fp_fn[s][s_val]["acc"] = (tp + tn) / (tp + tn + fp + fn)
            s_attr_to_fp_fn[s][s_val]["fscore"] = fscore
            s_attr_to_fp_fn[s][s_val]["recall"] = recall
            s_attr_to_fp_fn[s][s_val]["precision"] = precision


            if verbose == True:#
                if isinstance(s_val, float): # print the int value of the sensitive attr val
                    s_val = int(s_val)

                print ("+ |  %s  | %0.4f | %0.4f | %0.4f | %0.4f | %0.4f | %0.4f | %0.4f | %s | %0.0f | %0.0f | %0.0f | %0.0f " % (s_val,fpr,tpr,tp/L,acc,fscore, recall, precision,L  ,tp,tn,fp,fn ))
        
        if verbose == True:
           print("# | avg. | %0.4f | %0.4f | %0.4f | %0.4f | %0.4f  " % (
                                                 abs(s_attr_to_fp_fn[s][0]["fpr"] - s_attr_to_fp_fn[s][1]["fpr"]),
                                                 abs(s_attr_to_fp_fn[s][0]["tpr"] - s_attr_to_fp_fn[s][1]["tpr"]),
                                                 abs(s_attr_to_fp_fn[s][0]["tp"]-s_attr_to_fp_fn[s][1]["tp"]),
                                                 (s_attr_to_fp_fn[s][0]["acc"] + s_attr_to_fp_fn[s][1]["acc"] )/ 2,
                                                 (s_attr_to_fp_fn[s][0]["fscore"] + s_attr_to_fp_fn[s][1]["fscore"] ) / 2
                                                 ))

        return s_attr_to_fp_fn





