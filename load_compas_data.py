from __future__ import division
# import urllib2
import os, sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import feature_extraction
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer

from random import seed, shuffle

# sys.path.insert(0, '../../fair_classification/') # the code for fair classification is in this directory
import utilss as ut

from configparser import ConfigParser

SEED = 1234
seed(SEED)
np.random.seed(SEED)

"""
    The adult dataset can be obtained from: https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
    The code will look for the data file in the present directory, if it is not found, it will download them from GitHub.
"""


def check_data_file(fname):
    files = os.listdir(".")  # get the current directory listing
    print("Looking for file '%s' in the current directory..." % fname)

    if fname not in files:
        print("'%s' not found! Downloading from GitHub..." % fname)
        addr = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
        response = 1  # urllib2.urlopen(addr)
        data = response.read()
        fileOut = open(fname, "w")
        fileOut.write(data)
        fileOut.close()
        print("'%s' download and saved locally.." % fname)
    else:
        print("File found in current directory..")


def load_compas_data(input_File):
    config = ConfigParser()
    config.read('config.ini')

    COMPAS_INPUT_FILE = config.get(input_File, 'INPUT_FILE')  # "../../Data/AI_fairness/CSV_files/German-credit.csv"# "German-credit.csv"  # "compas-scores-two-years.csv"# "diabetic_sex.csv"
    # check_data_file(COMPAS_INPUT_FILE)

    # load the data and get some stats
    df = pd.read_csv(COMPAS_INPUT_FILE)
    df=df[df.Folds<10]
    df.sort_values('Folds',inplace=True)
    #df.drop('Folds', axis='columns', inplace=True)
    df.columns = df.columns.str.replace("'", '')

    # df = df.dropna(subset=["days_b_screening_arrest"]) # dropping missing vals

    #FEATURES_CLASSIFICATION = df.columns.drop(config.get(input_File, 'CLASS_FEATURE'))
    FEATURES_CLASSIFICATION =config.get(input_File, 'FEATURES_CLASSIFICATION').split(',')
    CONT_VARIABLES = config.get(input_File, 'CONT_VARIABLES').split(',')

    #  continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = config.get(input_File, 'CLASS_FEATURE')
    # "Y"# "two_year_recid"#'readmitted' #'c15'# the decision variable
    SENSITIVE_ATTRS = config.get(input_File, 'SENSITIVE_ATTRS').split()
    # ['Sex']
    try:
        df[SENSITIVE_ATTRS[0]][df[SENSITIVE_ATTRS[0]].str.contains('(?i)f|b|n')]=0# female, black, not married (bank DS)
        mask = ((df[SENSITIVE_ATTRS[0]] != 0) & (df[SENSITIVE_ATTRS[0]] != '0'))
        df[SENSITIVE_ATTRS[0]][mask] = 1
    except:
        df[SENSITIVE_ATTRS[0]][df[SENSITIVE_ATTRS[0]]==min(df[SENSITIVE_ATTRS[0]])] = 0 # female, black, not married (bank DS)
        mask = ((df[SENSITIVE_ATTRS[0]] != 0) & (df[SENSITIVE_ATTRS[0]] != '0'))
        df[SENSITIVE_ATTRS[0]][mask] = 1

    df[SENSITIVE_ATTRS[0]] = df[SENSITIVE_ATTRS[0]].astype(int)

    """ remove rows with few instances """
    if COMPAS_INPUT_FILE.split('/')[-1] in ['default of credit card clients_folds2.csv', 'cylinder.bands.csv']:
        for i in df[FEATURES_CLASSIFICATION].columns:
            continuse = all(map(str.isdigit, str(df[i])))
            if not continuse:
                df = df[df[i].map(df[i].value_counts()) >= 3]# works for ds 14 credit default
    # convert to np array
    data = df.to_dict('list')
    for k in data.keys():
        data[k] = np.array(data[k])

    """ Filtering the data """
    '''
    # These filters are the same as propublica (refer to https://github.com/propublica/compas-analysis)
    # If the charge date of a defendants Compas scored crime was not within 30 days from when the person was arrested, we assume that because of data quality reasons, that we do not have the right offense. 
    idx = np.logical_and(data["days_b_screening_arrest"]<=30, data["days_b_screening_arrest"]>=-30)


    # We coded the recidivist flag -- is_recid -- to be -1 if we could not find a compas case at all.
    idx = np.logical_and(idx, data["is_recid"] != -1)

    # In a similar vein, ordinary traffic offenses -- those with a c_charge_degree of 'O' -- will not result in Jail time are removed (only two of them).
    idx = np.logical_and(idx, data["c_charge_degree"] != "O") # F: felony, M: misconduct

    # We filtered the underlying data from Broward county to include only those rows representing people who had either recidivated in two years, or had at least two years outside of a correctional facility.
    idx = np.logical_and(idx, data["score_text"] != "NA")

    # we will only consider blacks and whites for this analysis
    #idx = np.logical_and(idx, np.logical_or(data["race"] == "African-American", data["race"] == "Caucasian"))
    idx = np.logical_or(data["race"] == "African-American", data["race"] == "Caucasian")

    # select the examples that satisfy this criteria
    for k in data.keys():
        data[k] = data[k][idx]
    '''

    """ Feature normalization and one hot encoding """

    # convert class label 0 to -1
    y = data[CLASS_FEATURE]
    y[y == 2] = -1
    y[y == 0] = -1
    y[y == "Low"] = "-1"
    y[y == "Fail"] =  "-1"
    y[y == "no"] =  "-1"
    y[y=="-50000"]=  "-1"
    y[y == "' <=50K'"] = "-1"
    y[y == 'N'] =  "-1"
    mask = ((y != -1) & (y != '-1'))
    y[mask] = 1
    y=y.astype(int)


    """ hstack the feature """

    X = np.array([]).reshape(len(y), 0)  # empty array with num rows same as num examples, will hstack the features to it
    x_control = defaultdict(list)

    feature_names = []
    for attr in FEATURES_CLASSIFICATION:
        vals = data[attr]
        continuse=all(map(str.isdigit,str(vals)))
        try:
            continuse=all(str(int(i)).isdigit() for i in vals)
        except:
            pass
        # add to sensitive features dict
        if attr in SENSITIVE_ATTRS:
            x_control[attr] = vals
            vals = np.reshape(vals, (len(y), -1))
        elif attr != 'Folds':
            if(continuse):
            #if attr in CONT_VARIABLES:
                vals = [float(v) for v in vals]
                vals2 = preprocessing.scale(vals)  # 0 mean and 1 variance
                vals2 = np.reshape(vals2, (len(y), -1))  # convert from 1-d arr to a 2-d arr with one col

                scaler = preprocessing.MinMaxScaler()
                #vals3 = preprocessing.MinMaxScaler(vals)  # 0 to 1 scaler
                vals3=np.reshape(vals,(-1, 1))
                vals3 = scaler.fit_transform(vals3)
                #X_test = scaler.transform(X_test)
                vals3 = np.reshape(vals3, (len(y), -1))  # convert from 1-d arr to a 2-d arr with one col

                bin = min(len(set(data[attr]))+1, 5)
                enc = KBinsDiscretizer(n_bins=bin, encode="ordinal")
                X_binned = enc.fit_transform(vals3)
                vals=X_binned


            else:  # for binary categorical variables, the label binarizer uses just one var instead of two
                lb = preprocessing.LabelEncoder()
                lb.fit(vals)
                vals = lb.transform(vals)
                '''
                vals=abs(vals)
    
                # vals = [abs(v) for v in vals]
                # vals4 = [abs(s) for s in vals if s.isdigit()]
                vals4=[]
                for v in vals:
                    if str(v).isdigit():
                        v=abs(v)
                    else:
                        v=v
                    vals4.append(v)
    
                '''
                vals = np.reshape(vals, (len(y), -1))  # convert from 1-d arr to a 2-d arr with one col

        else:
            vals = np.reshape(vals, (len(y), -1))  # convert from 1-d arr to a 2-d arr with one col



        # add to learnable features
        X = np.hstack((X, vals))


        feature_names.append(attr)

    # convert the sensitive feature to 1-d array
    x_control = dict(x_control)
    for k in x_control.keys():
        #assert (x_control[k].shape[1] == 1)  # make sure that the sensitive feature is binary after one hot encoding
        x_control[k] = np.array(x_control[k]).flatten()

    # sys.exit(1)

    """permute the date randomly
    perm = list(range(0, X.shape[0]))
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in x_control.keys():
        x_control[k] = x_control[k][perm]
    """
    #X = ut.add_intercept(X)

    #feature_names = ["intercept"] + feature_names
    assert (len(feature_names) == X.shape[1])
    #print("Features we will be using for classification are:", feature_names, "\n")
    # print input file after processing
    df2 = pd.DataFrame(X, columns=feature_names, dtype=float)
    df2['Y'] = y



    df2.to_csv(COMPAS_INPUT_FILE + '_processedNB2.csv', index=None)

    return X, y, x_control
