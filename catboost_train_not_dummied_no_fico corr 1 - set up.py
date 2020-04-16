# https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db

import numpy as np
from catboost.utils import get_gpu_device_count
import xlsxwriter
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import catboost as cb
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
#import pickle

print('I see %i GPU devices' % get_gpu_device_count())
start_time = time.time()

data = pd.read_excel(open('merged final data.xlsx', 'rb'),
                     sheet_name='Sheet1')
data = data.replace(['Fully paid', 'Charged Off'], [0, 1])
# data['CreditScore'] = np.log(data['CreditScore'])
data.drop(['CreditScore'], axis=1, inplace=True)

data.drop(['Avg_cur_bal'], axis=1, inplace=True)  # Tot_hi_cred_lim
data.drop(['Bc_util'], axis=1, inplace=True)  # Revol_util
data.drop(['Total_bc_limit'], axis=1, inplace=True)  # Total_rev_hi_lim

train, test, y_train, y_test = train_test_split(data.drop(["LoanStatus"], axis=1), data["LoanStatus"],
                                                random_state=10, test_size=0.30)
params = {'depth': [2, 6],
          'learning_rate': [0.1, 0.02],  # The learning rate. Used for reducing the gradient step.
          'l2_leaf_reg': [0.000001, 0.00001],  # support CPU and GPU
          'iterations': [500, 150,  # support CPU and GPU
          'rsm': [0.5, 0.6]# random subspace method, support CPU only
          #"rsm": [1]  # support GPU only
          }

cb1 = cb.CatBoostClassifier(
    # task_type="CPU",
    # border_count=254,# support GPU and CPU; for CPU:254. the numbler of splits for numerical features:1:65535
    # verbose=True,# GPU and CPU
    # devices=-1 # works, use all CPUs
    # #devices='0-10'# works too

    task_type="GPU",
    border_count=128,  # GPU:128
    verbose=True,
    devices='0-4'# for a range of devices (for example, devices='0-3')
    # devices='0:1'# for a range of devices (for example, devices='0-3')

)

cb_model = GridSearchCV(cb1, params, scoring="roc_auc", cv=5)
cb_model.fit(train, y_train)
print("--- %s seconds ---" % (time.time() - start_time))

from sklearn.metrics import accuracy_score

y_pred = cb_model.predict(test)
y_pred

# save the model to disk
import pickle

filename = 'finalized_CB_no_fico corr1.sav'
pickle.dump(cb_model, open(filename, 'wb'))

writer = pd.ExcelWriter('train and test data no fico corr1.xlsx', engine='xlsxwriter')
train.to_excel(writer, 'train', index=False)
test.to_excel(writer, 'test', index=False)
y_train.to_excel(writer, 'y_train', index=False)
y_test.to_excel(writer, 'y_test', index=False)
writer.save()

from sklearn.metrics import accuracy_score

y_pred = cb_model.predict_proba(test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred[:, 1])
print(metrics.auc(fpr, tpr))

yy_pred = y_pred[:, 1]
from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, yy_pred)
yy_pred[yy_pred >= 0.5] = 1
yy_pred[yy_pred < 0.5] = 0

accuracy_score(y_test, yy_pred)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, yy_pred)

y_pred = cb_model.predict_proba(test)
from sklearn.metrics import roc_curve

fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred[:, 1])

import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot([0, 1], [0, 1], 'b')
plt.plot(fpr_rt_lm, tpr_rt_lm, 'r', label='CatBbost model')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
