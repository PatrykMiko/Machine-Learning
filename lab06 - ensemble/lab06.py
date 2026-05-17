#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)


# In[2]:


from sklearn.model_selection import train_test_split

X = data_breast_cancer.data[['mean texture', 'mean symmetry']]
y = data_breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[3]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

dtr_clf = DecisionTreeClassifier()
log_clf = LogisticRegression(solver="lbfgs")
knn_clf = KNeighborsClassifier()

voting_clf_hard = VotingClassifier(
    estimators=[('lr', log_clf),
        ('dt', dtr_clf),
        ('knn', knn_clf)],
    voting='hard')

voting_clf_soft = VotingClassifier(
    estimators=[('lr', log_clf),
        ('dt', dtr_clf),
        ('knn', knn_clf)],
    voting='soft')

for clf in (dtr_clf, log_clf, knn_clf, voting_clf_hard, voting_clf_soft):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


# In[4]:


import pickle

dtr_clf.fit(X_train, y_train)
y_pred_train = dtr_clf.predict(X_train)
y_pred_test = dtr_clf.predict(X_test)
dtr_acc_train = accuracy_score(y_train, y_pred_train)
dtr_acc_test = accuracy_score(y_test, y_pred_test)

log_clf.fit(X_train, y_train)
y_pred_train = log_clf.predict(X_train)
y_pred_test = log_clf.predict(X_test)
log_acc_train = accuracy_score(y_train, y_pred_train)
log_acc_test = accuracy_score(y_test, y_pred_test)

knn_clf.fit(X_train, y_train)
y_pred_train = knn_clf.predict(X_train)
y_pred_test = knn_clf.predict(X_test)
knn_acc_train = accuracy_score(y_train, y_pred_train)
knn_acc_test = accuracy_score(y_test, y_pred_test)

voting_clf_hard.fit(X_train, y_train)
y_pred_train = voting_clf_hard.predict(X_train)
y_pred_test = voting_clf_hard.predict(X_test)
voting_hard_acc_train = accuracy_score(y_train, y_pred_train)
voting_hard_acc_test = accuracy_score(y_test, y_pred_test)

voting_clf_soft.fit(X_train, y_train)
y_pred_train = voting_clf_soft.predict(X_train)
y_pred_test = voting_clf_soft.predict(X_test)
voting_soft_acc_train = accuracy_score(y_train, y_pred_train)
voting_soft_acc_test = accuracy_score(y_test, y_pred_test)

acc_vote = [(dtr_acc_train, dtr_acc_test), (log_acc_train, log_acc_test), (knn_acc_train, knn_acc_test),
            (voting_hard_acc_train, voting_hard_acc_test), (voting_soft_acc_train, voting_soft_acc_test)]

with open('acc_vote.pkl', 'wb') as f:
    pickle.dump(acc_vote, f)

vote = [dtr_clf, log_clf, knn_clf, voting_clf_hard, voting_clf_soft]

with open('vote.pkl', 'wb') as f:
    pickle.dump(vote, f)


# In[5]:


from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=30,
    max_samples=1.0, bootstrap=True)
bag_clf.fit(X_train, y_train)
y_pred_train = bag_clf.predict(X_train)
y_pred_test = bag_clf.predict(X_test)
bag_acc_train = accuracy_score(y_train, y_pred_train)
bag_acc_test = accuracy_score(y_test, y_pred_test)
print(f"Bagging {bag_acc_test}")

bag_clf_half = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=30,
    max_samples=0.5, bootstrap=True)
bag_clf_half.fit(X_train, y_train)
y_pred_train = bag_clf_half.predict(X_train)
y_pred_test = bag_clf_half.predict(X_test)
bag_half_acc_train = accuracy_score(y_train, y_pred_train)
bag_half_acc_test = accuracy_score(y_test, y_pred_test)
print(f"Bagging z wykorzystaniem 50% instancji {bag_half_acc_test}")

past_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=30,
    max_samples=1.0, bootstrap=False)
past_clf.fit(X_train, y_train)
y_pred_train = past_clf.predict(X_train)
y_pred_test = past_clf.predict(X_test)
past_acc_train = accuracy_score(y_train, y_pred_train)
past_acc_test = accuracy_score(y_test, y_pred_test)
print(f"Pasting {past_acc_test}")

past_clf_half = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=30,
    max_samples=0.5, bootstrap=False)
past_clf_half.fit(X_train, y_train)
y_pred_train = past_clf_half.predict(X_train)
y_pred_test = past_clf_half.predict(X_test)
past_half_acc_train = accuracy_score(y_train, y_pred_train)
past_half_acc_test = accuracy_score(y_test, y_pred_test)
print(f"Pasting z wykorzystaniem 50% instancji {past_half_acc_test}")

rnd_clf = RandomForestClassifier(n_estimators=30)
rnd_clf.fit(X_train, y_train)
y_pred_train = rnd_clf.predict(X_train)
y_pred_test = rnd_clf.predict(X_test)
rnd_acc_train = accuracy_score(y_train, y_pred_train)
rnd_acc_test = accuracy_score(y_test, y_pred_test)
print(f"Random Forest {rnd_acc_test}")

ada_clf = AdaBoostClassifier(n_estimators=30)
ada_clf.fit(X_train, y_train)
y_pred_train = ada_clf.predict(X_train)
y_pred_test = ada_clf.predict(X_test)
ada_acc_train = accuracy_score(y_train, y_pred_train)
ada_acc_test = accuracy_score(y_test, y_pred_test)
print(f"AdaBoost {ada_acc_test}")

gbrt = GradientBoostingClassifier(n_estimators=30)
gbrt.fit(X_train, y_train)
y_pred_train = gbrt.predict(X_train)
y_pred_test = gbrt.predict(X_test)
gbrt_acc_train = accuracy_score(y_train, y_pred_train)
gbrt_acc_test = accuracy_score(y_test, y_pred_test)
print(f"Gradient Boosting {gbrt_acc_test}")

acc_bag = [(bag_acc_train, bag_acc_test), (bag_half_acc_train, bag_half_acc_test), (past_acc_train, past_acc_test),
            (past_half_acc_train, past_half_acc_test), (rnd_acc_train, rnd_acc_test), (ada_acc_train, ada_acc_test), (gbrt_acc_train, gbrt_acc_test)]

with open('acc_bag.pkl', 'wb') as f:
    pickle.dump(acc_bag, f)

bag = [bag_clf, bag_clf_half, past_clf, past_clf_half, rnd_clf, ada_clf, gbrt]

with open('bag.pkl', 'wb') as f:
    pickle.dump(bag, f)


# In[6]:


X = data_breast_cancer.data
y = data_breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

fea_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=30,
    max_samples=0.5, bootstrap=True, bootstrap_features=False,
    max_features=2)
fea_clf.fit(X_train, y_train)
y_pred_train = fea_clf.predict(X_train)
y_pred_test = fea_clf.predict(X_test)
fea_acc_train = accuracy_score(y_train, y_pred_train)
fea_acc_test = accuracy_score(y_test, y_pred_test)
print(f"Sampling 2 cech {fea_acc_test}")

acc_fea = [fea_acc_train, fea_acc_test]

with open('acc_fea.pkl', 'wb') as f:
    pickle.dump(acc_fea, f)

bag = [fea_clf]

with open('fea.pkl', 'wb') as f:
    pickle.dump(bag, f)


# In[7]:


import pandas as pd

train_accuracies = []
test_accuracies = []
features_names_list = []

for estimator, feature_indices in zip(fea_clf.estimators_, fea_clf.estimators_features_):

    selected_feature_names = list(X_train.columns[feature_indices])
    features_names_list.append(selected_feature_names)

    X_train_subset = X_train.iloc[:, feature_indices].values
    X_test_subset = X_test.iloc[:, feature_indices].values

    y_pred_train_est = estimator.predict(X_train_subset)
    y_pred_test_est = estimator.predict(X_test_subset)

    train_accuracies.append(accuracy_score(y_train, y_pred_train_est))
    test_accuracies.append(accuracy_score(y_test, y_pred_test_est))

df_rank = pd.DataFrame({
    'Train_Accuracy': train_accuracies,
    'Test_Accuracy': test_accuracies,
    'Feature_Names': features_names_list
})

df_rank_sorted = df_rank.sort_values(by=['Test_Accuracy', 'Train_Accuracy'], ascending=[False, False])

print(df_rank_sorted.head())
with open('acc_fea_rank.pkl', 'wb') as f:
    pickle.dump(df_rank_sorted, f)


# In[ ]:




