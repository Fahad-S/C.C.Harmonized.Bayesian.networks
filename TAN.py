import pandas as pd
import numpy as np
#import pgmpy
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#from pgmpy.models.BayesianModel import BayesianModel
from pgmpy.models.BayesianNetwork import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import TreeSearch, BayesianEstimator

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

# construct naive bayes graph and add interaction between the features #BayesianModel
model = BayesianNetwork([('C', 'R'), ('C', 'S'), ('C', 'T'), ('C', 'U'), ('C', 'V'),
                       ('R', 'S'), ('R', 'T'), ('R', 'U'), ('R', 'V')])
# add conditional probability distribution to edges
cpd_c = TabularCPD('C', 2, [[0.5], [0.5]])
cpd_r = TabularCPD('R', 3, [[0.6,0.2],[0.3,0.5],[0.1,0.3]], evidence=['C'],
                            evidence_card=[2])
cpd_s = TabularCPD('S', 3, [[0.1,0.1,0.2,0.2,0.7,0.1],
                            [0.1,0.3,0.1,0.2,0.1,0.2],
                            [0.8,0.6,0.7,0.6,0.2,0.7]],
                            evidence=['C','R'], evidence_card=[2,3])
cpd_t = TabularCPD('T', 2, [[0.7,0.2,0.2,0.5,0.1,0.3],
                            [0.3,0.8,0.8,0.5,0.9,0.7]],
                            evidence=['C','R'], evidence_card=[2,3])
cpd_u = TabularCPD('U', 3, [[0.3,0.8,0.2,0.8,0.4,0.7],
                            [0.4,0.1,0.4,0.1,0.1,0.1],
                            [0.3,0.1,0.4,0.1,0.5,0.2]],
                            evidence=['C','R'], evidence_card=[2,3])
cpd_v = TabularCPD('V', 2, [[0.5,0.6,0.6,0.5,0.5,0.4],
                            [0.5,0.4,0.4,0.5,0.5,0.6]],
                            evidence=['C','R'], evidence_card=[2,3])
model.add_cpds(cpd_c, cpd_r, cpd_s, cpd_t, cpd_u, cpd_v)

# generate sample data from our BN model
inference = BayesianModelSampling(model)
df_data = inference.forward_sample(size=30000)#, return_type='dataframe')

# split data into training and test set
cols_features = df_data.columns.tolist()
cols_features.remove('C')

X = df_data[cols_features].values
y = df_data[['C']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)
df_train = pd.DataFrame(np.concatenate((y_train, X_train), axis=1), columns=df_data.columns)

# train naive bayes classifier and predict
model_nb = MultinomialNB().fit(X_train, y_train)
y_pred = model_nb.predict(X_test)
print(classification_report(y_test, y_pred))

# learn the TAN graph structure from data
est = TreeSearch(df_train, root_node='U')
dag = est.estimate(estimator_type='tan', class_node='C')

# construct Bayesian network by parameterizing the graph structure
model = BayesianNetwork(dag.edges())
model.fit(df_train, estimator=BayesianEstimator, prior_type='K2')

# draw inference from BN
X_test_df = pd.DataFrame(X_test, columns=cols_features)
y_pred = model.predict(X_test_df).values
print(classification_report(y_test, y_pred))



