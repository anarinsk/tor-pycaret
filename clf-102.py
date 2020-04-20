# %% 
from pycaret.datasets import get_data
dataset = get_data('credit', profile=True)

# %%
data = dataset.sample(frac=0.95, random_state=786).reset_index(drop=True)
data_unseen = dataset.drop(data.index).reset_index(drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions ' + str(data_unseen.shape))

# %%
from pycaret.classification import *

# %%
exp_clf102 = setup(data = data, target = 'default', session_id=123,
                  normalize = True, 
                  transformation = True, 
                  ignore_low_variance = True,
                  remove_multicollinearity = True, multicollinearity_threshold = 0.95,
                  bin_numeric_features = ['LIMIT_BAL', 'AGE'],
                  group_features = [['BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'],
                                   ['PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']])

# %%
compare_models()
# %%
tuned_rf = tune_model('rf')
tuned_rf2 = tune_model('rf', optimize = 'AUC')

# %%
# lets create a simple decision tree model that we will use for ensembling 
dt = create_model('dt')

# %%
bagged_dt = ensemble_model(dt)

# %%
boosted_dt = ensemble_model(dt, method = 'Boosting')

# %%
tuned_bagged_dt = tune_model('dt', ensemble=True, method='Bagging')

# %%
