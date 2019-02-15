
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn


# In[3]:


housing = pd.read_csv('C:/Users/yl/Desktop/real_estate_analysis/datasets/housing.csv')


# In[4]:


housing.info()


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
housing.hist(bins=50, figsize = (20,15))
plt.show()


# In[7]:


np.random.seed(123)
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[: test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[8]:


train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), 'train +', len(test_set),'test')


# In[9]:


## preventing from adding training set into test set while updating the data
import hashlib
def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio
def split_train_test_by_id(data, test_ratio, id_col, hash = hashlib.md5):
    ids = data[id_col]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[-in_test_set], data.loc[in_test_set]


# In[10]:


# the housing dataset does not have an identifier column, use the row idx as ID
housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'index')


# In[11]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing_with_id, test_size = 0.2, random_state = 123)


# In[12]:


housing.median_income.hist()


# In[13]:


# genereate income categories
housing['income_cat'] = np.ceil(housing['median_income']/1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace = True)
housing['income_cat'].hist()
## most median income values are clustered around 20,000 - 50,000, but some go far beyond 60,000


# In[14]:


# stratified sampling based on the income category
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 123)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[15]:


#first take a look  at the income category proportions in the whole dataset
housing['income_cat'].value_counts()/len(housing)


# In[16]:


strat_test_set['income_cat'].value_counts()/len(strat_test_set)


# In[19]:


view_dif = pd.DataFrame(housing['income_cat'].value_counts()/len(housing))
#view_dif['random_split'] = test_set['income_cat'].value_counts()/len(test_set)
view_dif['strat_split'] = strat_test_set['income_cat'].value_counts()/len(strat_test_set)
print(view_dif) 
## strat_split approximately equal to the whole data set while random split has some skewness


# In[20]:


#remove income cat column
for rm in (strat_train_set, strat_test_set):
    rm.drop('income_cat', axis = 1, inplace = True)


# In[21]:


# pre-test
housing = strat_train_set.copy()


# In[22]:


# since geo info, go with scatter
housing.plot(kind='scatter', x = 'longitude', y = 'latitude')  # look like the shape of California?


# In[23]:


# check out some high-density area
housing.plot(kind='scatter', x = 'longitude', y = 'latitude', alpha = 0.1)
# SF, LA, San Diego basically


# In[24]:


# look at housing price, the radius of each cricle represents the district's population, the color represents the price
housing.plot(kind = 'scatter', x = 'longitude', y = 'latitude', alpha = 0.4, 
            s = housing['population']/100, label = 'population', figsize = (10, 7),
            c = 'median_house_value', cmap = plt.get_cmap('jet'), colorbar = True)
plt.legend()
# the img reveals that the housing prices are very much related to the location and to the population density


# In[25]:


import matplotlib.image as mlimg
cali_img = mlimg.imread('C:/Users/yl/Desktop/real_estate_analysis/datasets/california.png')
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                       s=housing['population']/100, label="Population",
                       c="median_house_value", cmap=plt.get_cmap("jet"),
                       colorbar=False, alpha=0.4,
                      )
plt.imshow(cali_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar()
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
#save_fig("california_housing_prices_plot")
plt.show()


# In[26]:


corr_mat = housing.corr()
print(corr_mat['median_house_value'].sort_values(ascending = False))


# In[27]:


# correlation plot with promising attributes
from pandas.tools.plotting import scatter_matrix
atts = ['median_house_value', 'median_income','total_rooms','housing_median_age']
scatter_matrix(housing[atts], figsize = (12,8))


# In[28]:


# try with the most promising attributes --- median income
housing.plot(kind ='scatter', x= 'median_income', y = 'median_house_value', alpha = 0.1)


# In[29]:


housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['population_per_household'] = housing['population']/housing['households']


# In[30]:


corr_mat = housing.corr()
corr_mat['median_house_value'].sort_values(ascending = False)


# In[31]:


# data preparation
housing = strat_train_set.drop('median_house_value', axis = 1)
housing_labels = strat_train_set['median_house_value'].copy()


# In[32]:


sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows
# reminds of bedrooms_per_room col(so as bedrooms_per_room) has only 20433 obs rather than 20640, so there is some missing data


# In[34]:


# data cleaning
## filling the missing value with median value of that column
#housing['total_bedrooms'].fillna(housing['total_bedrooms'].median(), inplace = True)
# or go with Imputer in Sklearn
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')
#since the imputer can only be computed on numerical attributes, so need to create a copy of data without ocean proximity
housing_num = housing.drop('ocean_proximity', axis = 1)


# In[35]:


imputer.fit(housing_num)


# In[36]:


imputer.statistics_


# In[37]:


housing_num.median().values


# In[38]:


X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns = housing_num.columns, index = list(housing.index.values))


# In[39]:


housing_tr.info()


# In[42]:


# since the ocean proximity is a text attribute, so we cannot compute its medians. 
# Therefore, I implement LabelEncoder to transfer it to be numerical
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
housing_cat = housing['ocean_proximity']
housing_cat_encoded = encoder.fit_transform(housing_cat.reshape(-1,1))
print(housing_cat_encoded)
#print(encoder.classes_) # label 0,4 are more similiar than 0,1, but ML algo will assume that two nearby values are more similiar


# In[43]:


encoder.categories_


# In[46]:


# try one hot encoder, which transform the cat into binary numbers
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_tr_1hot = cat_encoder.fit_transform(housing_cat.reshape(-1,1))
housing_tr_1hot[:10].toarray()


# In[149]:


housing_tr_1hot.toarray()


# In[47]:


# or we can apply both transformation in one shot
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
tmp = encoder.fit_transform(housing['ocean_proximity']) #sparse_output = False by default
print(tmp)


# In[48]:


housing.columns


# In[49]:


# try to customize a transformer
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")]

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y = None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:,rooms_idx]/X[:,household_idx]
        population_per_household = X[:, population_idx]/X[:, household_idx]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_idx]/X[:,rooms_idx]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# In[50]:


from sklearn.preprocessing import FunctionTransformer

def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
                     bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]

attr_adder = FunctionTransformer(add_extra_features, validate=False,
                                 kw_args={"add_bedrooms_per_room": False})
housing_extra_attribs = attr_adder.fit_transform(housing.values)


# In[51]:


housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"])
housing_extra_attribs.head()


# In[152]:


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attrs = attr_adder.transform(housing.values)


# In[153]:


#alternatively we can use FunctionTransformer
from sklearn.preprocessing import FunctionTransformer

def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
                     bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]

attr_adder = FunctionTransformer(add_extra_features, validate=False,
                                 kw_args={"add_bedrooms_per_room": False})
housing_extra_attribs = attr_adder.fit_transform(housing.values)


# In[154]:


housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"])
housing_extra_attribs.head()


# In[53]:


# feature scaling
## normalization or standardization
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
    ('std_scaler', StandardScaler()),
])
housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[54]:


housing_num_tr


# In[55]:


from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)


# In[56]:


housing_prepared


# In[57]:


housing_prepared.shape


# In[59]:


#SELECT and Train A model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[60]:


# try it out with some instances from the training set
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prep = full_pipeline.transform(some_data)
print('Predictions:',lin_reg.predict(some_data_prep))
print('Labels:', list(some_labels))


# In[61]:


# measure accuracy with RMSE
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[ ]:


######### linear regression performs bad!!!!!!!! UNDERFIT! SO try some more powerful models


# In[62]:


## try decision tree
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# In[63]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_predictions, housing_labels)
tree_rmse = np.sqrt(tree_mse)
tree_rmse ## ???? 0.0 deflt overfit


# In[64]:


### evaluate with cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                        scoring = 'neg_mean_squared_error', cv = 10)
tree_rmse_scores = np.sqrt(-scores)


# In[65]:


print('Scores:' ,tree_rmse_scores)
print('Mean:', tree_rmse_scores.mean())
print('Std:', tree_rmse_scores.std()) ####### looks like worse than linear regression


# In[66]:


## try ag with lr
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring = 'neg_mean_squared_error', cv = 10)
lin_rmse_scores = np.sqrt(-lin_scores)
print('Scores:' ,lin_rmse_scores)
print('Mean:', lin_rmse_scores.mean())
print('Std:', lin_rmse_scores.std()) ####### looks like better than DeciTree, which means DeciTree is badly overfitting


# In[67]:


## try random forest
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
forest_mse = mean_squared_error(forest_reg.predict(housing_prepared),housing_labels)
forest_rmse = np.sqrt(forest_mse)
forest_rmse ##### not that bad


# In[68]:


forest_scores = cross_val_score(forest_reg,housing_prepared, housing_labels, scoring = 'neg_mean_squared_error', cv = 10)
forest_rmse_scores = np.sqrt(-forest_scores)
print('Scores:' ,forest_rmse_scores)
print('Mean:', forest_rmse_scores.mean()) 
print('Std:', forest_rmse_scores.std()) ## better than DeciTree and lr so far


# In[69]:


####Fine tune the model
##grid search to fiddle with the hyperparams
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [3,10,30], 'max_features': [2,4,6,8]},
    {'bootstrap':[False], 'n_estimators':[3,10], 'max_features': [2,3,4]}
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv = 5, scoring = 'neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)


# In[70]:


grid_search.best_params_


# In[71]:


grid_search.best_estimator_


# In[72]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params) ### so go with 6 and 30 as hyperparams


# In[73]:


##### analyze the best models and their errors
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[76]:


### evaluate the model on the test set
from sklearn.metrics import mean_squared_error
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop('median_house_value', axis = 1)
y_test = strat_test_set['median_house_value'].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predicitions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predicitions)
final_rmse = np.sqrt(final_mse)
print(final_rmse) ## ===> 49288


# In[78]:


from sklearn.svm import SVR
svm_reg = SVR(kernel = 'linear')
svm_reg.fit(housing_prepared, housing_labels)
svm_pred = svm_reg.predict(X_test_prepared)
svm_mse = mean_squared_error(y_test, svm_pred)
print(np.sqrt(svm_mse))

