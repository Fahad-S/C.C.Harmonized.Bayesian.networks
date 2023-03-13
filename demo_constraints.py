import os,sys
import numpy as np
from load_compas_data import *
#sys.path.insert(0, '/') # the code for fair classification is in this directory
import utilss as ut
import funcs_disp_mist as fdm
from copy import deepcopy

import pyAgrum.lib.notebook as gnb

from configparser import ConfigParser
import warnings
warnings.filterwarnings('ignore')#, '.*do not.*', )

def test_compas_data(input_File,pre_processing):
    config = ConfigParser()
    config.read('config.ini')
    file = config.get(input_File, 'INPUT_FILE')

    """ Generate the synthetic data """
    data_type = 1
    X, y, x_control = load_compas_data(input_File)
    sensitive_attrs = x_control.keys()
    # get the index of 'dog'
    index = config.get(input_File, 'FEATURES_CLASSIFICATION').split(',').index(list(sensitive_attrs)[0])

    if(pre_processing):

        one_one = np.sum(np.logical_and(y == 1,x_control[config.get(input_File,'SENSITIVE_ATTRS')] == 1))  # something which is -ve but is misclassified as +ve
        one_zero = np.sum(np.logical_and(y == 1, x_control[config.get(input_File, 'SENSITIVE_ATTRS')] == 0))  # something which is -ve but is misclassified as +ve
        zero_one = np.sum(np.logical_and(y == -1, x_control[config.get(input_File, 'SENSITIVE_ATTRS')] == 1))  # something which is -ve but is misclassified as +ve
        zero_zero = np.sum(np.logical_and(y == -1, x_control[config.get(input_File, 'SENSITIVE_ATTRS')] == 0))  # something which is -ve but is misclassified as +ve
        tot=one_one+one_zero+zero_one+zero_zero

        print(str(file).split('/')[-1],'|',len(y),'|',len(X[0]),'|', sensitive_attrs,'|',one_one,'|',one_zero,'|',zero_one,'|',zero_zero,'|',tot)
        return


    """ Split the data into train and test """

    print('main | ' + str(fileNo),  ' | file | ', file)

    for fold in range(train_fold_size):

        '''
            2NB implementation: you can remove this nested for part and uncomment the true line for split, then back indent def train.. fucnction() twice !!!!

       
        x_control_internal = deepcopy(x_control)
        for s in sensitive_attrs:
            s_attr_vals = x_control_internal[s]
            for s_val in sorted(list(set(s_attr_vals))):
                X_local = X[s_attr_vals == s_val]
                y_local = y[s_attr_vals == s_val]
                x_control_local={}
                x_control_local[s]=x_control_internal[s][s_attr_vals == s_val]

                x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X_local, y_local,x_control_local, train_fold_size,fold)
        '''
        x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, train_fold_size, fold)
        if fold<5:
            print('fold | ', fold)

            def train_test_classifier(NB):

                """ init. classifier"""
                w = fdm.train_model_disp_mist(x_train, y_train, x_control_train, NB,index)

                ''' Laplace '''
                # α/(N + α⋅k) --> N: p(c)*size == len(C=c),  k: |ai| # values for ai [differ per att]
                n_idx = 0
                Prob_term =0
                for i in w.cpds:
                    if i.variable == 'col_Y':
                        Prob_term = w.cpds[n_idx].values
                        w.cpds[n_idx].values[0] += 1 / (w.cpds[n_idx].values[0] * len(x_train) + 2)
                        w.cpds[n_idx].values[1] += 1 / (w.cpds[n_idx].values[1] * len(x_train) + 2)
                    n_idx += 1
                n_idx = 0
                for i in w.cpds:
                    if len(i.cardinality) < 3 and i.variable != 'col_Y':
                        for z_idx in range(len(w.cpds[n_idx].values)):
                            w.cpds[n_idx].values[z_idx][0] += 1 / (Prob_term[0] * len(x_train) + i.cardinality[0])
                            w.cpds[n_idx].values[z_idx][1] += 1 / (Prob_term[1] * len(x_train) + i.cardinality[0])

                    elif len(i.cardinality) >= 3 and i.variable != 'col_Y':
                        for z_idx in range(len(w.cpds[n_idx].values)):
                            w.cpds[n_idx].values[z_idx][0][0] += 1 / (Prob_term[0] * len(x_train) + i.cardinality[0])
                            w.cpds[n_idx].values[z_idx][0][1] += 1 / (Prob_term[1] * len(x_train) + i.cardinality[0])
                    n_idx += 1

                n_idx=0
                for i in w.cpds:
                    softmax = w.cpds[n_idx].values.sum(axis=0).flatten()
                    if i.variables[0] == 'col_Y':
                        w.cpds[n_idx].values[0] = w.cpds[n_idx].values[0] / softmax
                        w.cpds[n_idx].values[1] = w.cpds[n_idx].values[1] / softmax
                    elif len(i.cardinality) < 3:
                        for z_idx in range(len(w.cpds[n_idx].values)):
                            w.cpds[n_idx].values[z_idx][0] = w.cpds[n_idx].values[z_idx][0] / softmax[0]
                            w.cpds[n_idx].values[z_idx][1] = w.cpds[n_idx].values[z_idx][1] / softmax[1]
                    elif len(i.cardinality) >= 3:
                        for z_idx in range(len(w.cpds[n_idx].values)):
                            w.cpds[n_idx].values[z_idx][0][0] = w.cpds[n_idx].values[z_idx][0][0] / softmax[0]
                            w.cpds[n_idx].values[z_idx][0][1] = w.cpds[n_idx].values[z_idx][0][1] / softmax[1]

                    n_idx += 1

                """ test init. classifier"""
                # print test stats.
                train_score, test_score, s_attr_to_fp_fn_train, s_attr_to_fp_fn_test = fdm.get_clf_stats(w, x_train,y_train,x_control_train,x_test, y_test,x_control_test,sensitive_attrs, NB)
                '''
                #  check if training stats. improved after fine-tuning
                # L1 = s_attr_to_fp_fn_train[list(sensitive_attrs)[0]][0]["L"]
                # L2 = s_attr_to_fp_fn_train[list(sensitive_attrs)[0]][1]["L"]
                # tp1 = s_attr_to_fp_fn_train[list(sensitive_attrs)[0]][0]["tp"] / L1  # * L1 / (L1 + L2)
                # tp2 = s_attr_to_fp_fn_train[list(sensitive_attrs)[0]][1]["tp"] / L2  # * L2 / (L1 + L2)
                fpr1 = s_attr_to_fp_fn_train[list(sensitive_attrs)[0]][0]["mcc"]#tpr
                fpr2 = s_attr_to_fp_fn_train[list(sensitive_attrs)[0]][1]["mcc"]
                tpD = 1 - abs(fpr1 - fpr2)
                '''
                Acc1 = s_attr_to_fp_fn_train[list(sensitive_attrs)[0]][0]["mcc"]  # * L1 / (L1 + L2)
                Acc2 = s_attr_to_fp_fn_train[list(sensitive_attrs)[0]][1]["mcc"]  # * L2 / (L1 + L2)

                Acc = 2 / (1 / max(Acc1, 1e-25) + 1 / max(Acc2, 1e-25))
                # AccD = max(1 - (abs(Acc1 - Acc2) / 2), 1e-5)
                term = Acc  # 2 / (1 / Acc + 1 / AccD)

                best_term = term
                iterationNo=0
                improved=False
                worse_than = 0
                best_iter=0
                w_init = w.copy() # similar to best = 0 and get init. NB as best

                print("Term:%0.4f | Best_Term:%0.4f | itr:%0.0f | %0.4f" % (
                term, best_term, iterationNo, Acc))

                if(NB>=2):
                    """  fine tuning loop """
                    while (iterationNo<maxIter and worse_than<=maxWorst):

                        iterationNo+=1
                        acc = (y_train == np.array(train_score))
                        miss = 0
                        ''' loop on training data and if misclassified update instance  prob. terms w/ harmonic average '''
                        for inst_idx in range (len(x_train)):
                            '''
                            #Incremental
                            df_pred = pd.DataFrame(np.array(x_train[inst_idx]).reshape(1, len(x_train[inst_idx])))
                            df_pred.rename(lambda x: 'col_' + str(x), axis='columns', inplace=True)
                            prediction_inst = w.predict(df_pred)
                            prediction_inst = prediction_inst['col_Y']
                            acc = (y_train[inst_idx] == np.array(prediction_inst))
                            '''
                            if(not acc[inst_idx]):  # misclassified
                                miss += 1
                                n_idx=0

                                from random import randrange
                                threshold=2
                                if fileNo==1 or fileNo==2:
                                    threshold=3

                                if randrange(10)<threshold:

                                    for i in w.cpds:


                                        if (y_train[inst_idx] == 1):  # other value = -1 but index will be 0
                                            actual_idx = 1
                                        else:
                                            actual_idx = 0
                                        if i.variables[0]=='col_Y':
                                            v_idx =-1
                                        else:
                                            v_idx = int(str.replace(i.variables[0], 'col_', ''))  # get att name/index
                                        z_idx=int(x_train[inst_idx, v_idx]) # get att/column value in misclassfied training instance, to be fine-tuned in root/first node (only root+class)

                                        if i.variables[0]=='col_Y':
                                            term_prob = w.cpds[n_idx].values # both class values
                                            actual = term_prob[actual_idx]  # check actual vs predicted classes, since its misclassified
                                            predicted = term_prob[1 - actual_idx]
                                            HarmoniProbPartial = 1 - (2 / ((1 / actual) + (1 / predicted)))  # Term Prob
                                            newProbVal = abs(lr * abs(HarmoniProbPartial) / iterationNo)#pow(iterationNo, 1.0))

                                            #if actual < predicted:
                                            w.cpds[n_idx].values[actual_idx] = max(min(w.cpds[n_idx].values[actual_idx] + newProbVal, 1.0), 1e-50)
                                            #if actual < predicted:
                                            w.cpds[n_idx].values[1 - actual_idx] = max(min(w.cpds[n_idx].values[1 - actual_idx] - newProbVal, 1.0), 1e-50)
                                            ''' softmax '''
                                            softmax = w.cpds[n_idx].values.sum(axis=0).flatten()
                                            w.cpds[n_idx].values[0] = w.cpds[n_idx].values[0] / softmax
                                            w.cpds[n_idx].values[1] = w.cpds[n_idx].values[1] / softmax

                                        elif len(i.cardinality) == 2: # parent node (class in NB and other parent in BN)
                                            if int(str.replace(i.variables[0], 'col_', '')) != -1: #  == index: #!= -1: # # sens. att update only
                                                term_prob = w.cpds[n_idx].values[z_idx]  # both class values
                                                ''' now check actual vs predicted classes, since its misclassified '''
                                                actual = term_prob[actual_idx]
                                                predicted = term_prob[1 - actual_idx]
                                                HarmoniProbPartial = 1 - (2 / ((1 / actual) + (1 / predicted)))  # Term Prob
                                                newProbVal = abs(lr * abs(HarmoniProbPartial) / iterationNo)#pow(iterationNo, 1.0))

                                                w.cpds[n_idx].values[z_idx][actual_idx] = max(min(w.cpds[n_idx].values[z_idx][actual_idx] + newProbVal, 1.0),1e-50)
                                                #if actual < predicted:
                                                w.cpds[n_idx].values[z_idx][1 - actual_idx] = max(min(w.cpds[n_idx].values[z_idx][1 - actual_idx] - newProbVal, 1.0),1e-50)
                                                ''' softmax '''
                                                softmax = w.cpds[n_idx].values.sum(axis=0).flatten()
                                                for k in range(i.cardinality[0]):
                                                    w.cpds[n_idx].values[k][actual_idx] = w.cpds[n_idx].values[k][actual_idx] / softmax[int(actual_idx)]
                                                    w.cpds[n_idx].values[k][1 - actual_idx] = w.cpds[n_idx].values[k][1 - actual_idx] / softmax[int(1 - actual_idx)]

                                        elif len(i.cardinality)==3:
                                            #print('fuck..... it BN ...')
                                            #if int(str.replace(i.variables[0], 'col_', '')) != -1: #== index: # sens. att update only
                                            if int(str.replace(i.variables[0], 'col_', '')) == index : #or (int(str.replace(i.variables[1], 'col_', '')) == index)):
                                                term_prob = w.cpds[n_idx].values[z_idx][0]  # both class values
                                                ''' now check actual vs predicted classes,, since its misclassified '''
                                                actual = term_prob[actual_idx]
                                                predicted = term_prob[1 - actual_idx]
                                                HarmoniProbPartial = 1 - (2 / ((1 / actual) + (1 / predicted)))  # Term Prob
                                                newProbVal = abs(lr * abs(HarmoniProbPartial) / iterationNo)# pow(iterationNo, 1.0))

                                                w.cpds[n_idx].values[z_idx][0][actual_idx] = max(min(w.cpds[n_idx].values[z_idx][0][actual_idx] + newProbVal, 1.0),1e-50)
                                                #if actual < predicted:
                                                w.cpds[n_idx].values[z_idx][0][1 - actual_idx]= max(min(w.cpds[n_idx].values[z_idx][0][1- actual_idx] - newProbVal, 1.0), 1e-50)
                                                ''' softmax '''
                                                softmax = w.cpds[n_idx].values.sum(axis=0).flatten()
                                                for k in range(i.cardinality[0]):
                                                    w.cpds[n_idx].values[k][0][actual_idx] = w.cpds[n_idx].values[k][0][actual_idx] / softmax[int(actual_idx)]
                                                    w.cpds[n_idx].values[k][0][1 - actual_idx] = w.cpds[n_idx].values[k][0][1 - actual_idx] / softmax[int(1 - actual_idx)]
                                        else:
                                            fuck='me'
                                        n_idx+=1
                                    # End of Att.
                                # End of random

                            #End of miss


                        # End of isnts.
                        """     Batch training      """
                        """ End of fine-tuning epoch on training dataset, now let's print the modified model stats. again for testing dataset """
                        train_score, test_score, s_attr_to_fp_fn_train, s_attr_to_fp_fn_test = fdm.get_clf_stats(w, x_train,  y_train,x_control_train,x_test, y_test,x_control_test, sensitive_attrs, 4)

                        #  check if training stats. improved after fine-tuning
                        '''
                        L1 = s_attr_to_fp_fn_train[list(sensitive_attrs)[0]][0]["L"]
                        L2 = s_attr_to_fp_fn_train[list(sensitive_attrs)[0]][1]["L"]
                        fpr1 = s_attr_to_fp_fn_train[list(sensitive_attrs)[0]][0]["fpr"]
                        fpr2 = s_attr_to_fp_fn_train[list(sensitive_attrs)[0]][1]["fpr"]
                        tpr1 = s_attr_to_fp_fn_train[list(sensitive_attrs)[0]][0]["tpr"]
                        tpr2 = s_attr_to_fp_fn_train[list(sensitive_attrs)[0]][1]["tpr"]
                        tp1 = s_attr_to_fp_fn_train[list(sensitive_attrs)[0]][0]["tp"] / L1  # * L1 / (L1 + L2)
                        tp2 = s_attr_to_fp_fn_train[list(sensitive_attrs)[0]][1]["tp"] / L2  # * L2 / (L1 + L2)
                        mcc1 = s_attr_to_fp_fn_train[list(sensitive_attrs)[0]][0]["mcc"]
                        mcc2 = s_attr_to_fp_fn_train[list(sensitive_attrs)[0]][1]["mcc"]
                        tpD = (1 - abs(fpr1 - fpr2) + 1 - abs(tpr1 - tpr2) + 1 - abs(tp1 - tp2))/3
                        '''
                        Acc1 = s_attr_to_fp_fn_train[list(sensitive_attrs)[0]][0]["mcc"]  # * L1 / (L1 + L2)
                        Acc2 = s_attr_to_fp_fn_train[list(sensitive_attrs)[0]][1]["mcc"]  # * L2 / (L1 + L2)


                        Acc = 2 / (1 / max(Acc1, 1e-25) + 1 / max(Acc2, 1e-25))
                        #AccD = max(1 - (abs(Acc1 - Acc2) / 2), 1e-5)
                        term = Acc#2 / (1 / Acc + 1 / AccD)

                        if (term > best_term):
                            # print("Better:",Term, best_term, "-> itr.",iterationNo)
                            best_term=term
                            improved=True
                            w_bck = w.copy()
                            worse_than=0
                            best_iter=iterationNo
                            #print("Term:%0.4f | Best_Term:%0.4f | itr:%0.0f | %0.4f | %0.4f" % (term, best_term, iterationNo, 1 - tpD, Acc))
                        elif(term <= best_term):
                            worse_than+=1



                    """ End of iterations fine-tuning process  """
                    if improved:
                        train_score, test_score, s_attr_to_fp_fn_train, s_attr_to_fp_fn_test = fdm.get_clf_stats(w_bck, x_train,y_train, x_control_train,x_test, y_test,x_control_test,sensitive_attrs,NB)
                    else:
                        train_score, test_score, s_attr_to_fp_fn_train, s_attr_to_fp_fn_test = fdm.get_clf_stats(w_init, x_train,y_train, x_control_train, x_test,y_test, x_control_test,sensitive_attrs,NB)



                    print(iterationNo, improved, best_iter, worse_than)
                # just in case you need it. here is accuracy and FPR are for the test because we need of for plotting
                return w, test_score, s_attr_to_fp_fn_test


            """ Classify the data while optimizing for accuracy """
            #print ("== NB (original) classifier ==")
            #w_uncons, acc_uncons, s_attr_to_fp_fn_test_uncons = train_test_classifier(1)

            """ Now classify such that we optimize for Fscore while achieving better fairness """
            print("-- 3D-NB classifier --")
            model1, acc_uncons, s_attr_to_fp_fn_test_uncons1 = train_test_classifier(2)

            #print("++ 3D-BN classifier +++")
            #model2, acc_uncons, s_attr_to_fp_fn_test_uncons2 = train_test_classifier(3)



    return


for fileNo in range(1,12):#5  15
    #print (i)
    pre_processing=False#True
    #if i+1 in [1,2,3,7,8,9,14]: continue
    maxIter = 100
    maxWorst=6
    # lr = 0.01 good for student, 1
    #if fileNo==2:
    #lr = 0.001  # good for G credit, 2
    #elif fileNo==3 or fileNo==4 or fileNo==10:
    #    lr = 0.00005 # for propaublica 3,4
    #elif fileNo==9:
    #    lr = 0.0001 # for adult-race 9

    lr = 1e-3
    #if fileNo>2:#4
    #    lr = 1e-4
    train_fold_size = 10
    if fileNo == 1 or fileNo ==2 or fileNo == 9:
        train_fold_size = 5



    test_compas_data('main'+str(fileNo), pre_processing)

