"""
python code for training and evaluating the performance of the SVM classifier
for the loan3b.csv dataset.

The data is shuffled to avoid biasing.

When considering 30% of the whole dataset -
    Here 70% is taken as training data and remaining 30% is test data
    Classification Accuracy = 0.791611956438
    Classification Error = 0.208388043562

For the Complete data -
    Here 70% is taken as training data and remaining 30% is test data
    we have to comment the line in the code "lc = lc[:(int(lc.shape[0]*0.30))]"
    Classification Accuracy = 0.935534700498
    Classification Error = 0.0644652995018
"""

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import svm
from sklearn.metrics import zero_one_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

# DATA PRE PROCESSING

# Reading the data using pandas library
lc = pd.read_csv("loan3b.csv")
lc['term'] = lc['term'].fillna("60")
lc.term = pd.Series(lc.term).str.replace(' months', '').astype(int)
lc = lc[lc.term == 36]

# Using 30% of the total data
#lc = lc[:(int(lc.shape[0]*0.30))]
# commented to use the full data set

# describing the data. Used to show the numerical values
lc.describe()

# deleting the data that does not provide any information
del lc['id']
del lc['member_id']
del lc['url']
del lc['desc']
del lc['title']

# Looking at column values  to decide if irrelevant columns are present
print lc.columns.values

# dropping the columns that does not provide any information
lc.drop(['emp_title', 'issue_d','accept_d','exp_d','list_d','sub_grade','emp_title','pymnt_plan'],1,inplace = 'True')

# Converting the 3% to numerical 3
lc.int_rate = pd.Series(lc.int_rate).str.replace('%', '').astype(float)
lc.revol_util = pd.Series(lc.revol_util).str.replace('%', '').astype(float)

# Non numeric values are replaced.
lc.replace('n/a', np.nan,inplace=True)
lc.emp_length.fillna(value=0,inplace=True)
lc['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
lc['emp_length'] = lc['emp_length'].astype(int)

# Fully Paid feature is classified as 1 So mapped to number 1
# Otherwise,the feature (observation) is classified as 0
# print(lc["loan_status"].unique())
lc.loc[lc["loan_status"] != "Fully Paid","loan_status"] = 0
lc.loc[lc["loan_status"] == "Fully Paid", "loan_status"] = 1


# changing the date into number of days to be included in the model
lc.earliest_cr_line = pd.to_datetime(lc.earliest_cr_line)
date = datetime.now().strftime('%Y-%m-%d')
lc.earliest_cr_line = lc.earliest_cr_line.apply(lambda x: (np.timedelta64((x - pd.Timestamp(date)),'D').astype(int))/-365)



# calculating the mean FICO
lc['meanfico'] = (lc.fico_range_low + lc.fico_range_high)/2
# Dropping the ranges as its not needed now
lc.drop(['fico_range_low','fico_range_high','initial_list_status'],1, inplace=True)

# calculating the mean last fico of borrower's pulled
# lc['last_fico_range'] = lc.last_fico_range_low.astype('str') + '-' + lc.last_fico_range_high.astype('str')
lc['last_meanfico'] = (lc.last_fico_range_low + lc.last_fico_range_high)/2
# Dropping the ranges as its not needed now
lc.drop(['last_fico_range_high','last_fico_range_low','policy_code'],1, inplace=True)

# Dropping the columns that does not contribute
lc.drop(['out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp','addr_city',
         'total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee','last_pymnt_d','last_pymnt_amnt','grade','addr_state',
         'next_pymnt_d','last_credit_pull_d','funded_amnt','funded_amnt_inv', 'installment', 'mths_since_last_delinq', 'total_acc'],1, inplace=True)

# Replacing the null values
lc.fillna(0.0,inplace=True)
lc.fillna(0,inplace=True)

# Information about the dataFrame after removing irrelevant informationi
print lc.info()

# Changing the Income Verified column and feature need columns, which currently has textual labels to numeric.
le=LabelEncoder()
lc.is_inc_v = le.fit_transform(lc.is_inc_v.values)
lc.home_ownership=le.fit_transform(lc.home_ownership.values)
lc.purpose=le.fit_transform(lc.purpose.values)
# converting all fields to numeric
lc=lc.convert_objects(convert_numeric=True)

# drop highly correlated/redundant data to address multicollinearity
cor = lc.corr()
cor.loc[:,:] = np.tril(cor, k=-1) # below main lower triangle of an array
cor = cor.stack()
print cor[(cor > 0.55) | (cor < -0.55)]

# Using the above we remove the correlated/redundant data
lc.drop(['percent_bc_gt_75','bc_util','total_bc_limit','total_il_high_credit_limit','mths_since_recent_bc_dlq','pub_rec_bankruptcies',
         'tax_liens','num_sats','tot_cur_bal','avg_cur_bal','num_bc_tl','mo_sin_old_rev_tl_op','num_actv_rev_tl'
            ,'num_rev_tl_bal_gt_0'], 1, inplace = 'True')


# USING THE SVC CLASSIFIER FROM SKLEARN

y = lc.loan_status.values
# delecting the column from the input data
del lc['loan_status']

X = lc.values

training_data = int(X.shape[0] * 0.70) #Considering last 30# as Test data
X_train, y_train = X[:training_data], y[:training_data]
X_test, y_test = X[training_data:], y[training_data:]


# Fitting the Classifier model
model = svm.SVC(kernel='rbf',C=10000.0)  # can check with different kernels and C
model.fit(X_train, y_train)

print "Accuracy = ",model.score(X_test,y_test)

#Predict Output
y_pred = model.predict(X_test)
error_rate = zero_one_loss(y_test, y_pred)
print "Error rate =", error_rate

