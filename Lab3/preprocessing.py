import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("kc_house_data.csv", parse_dates = ['date']) # load the data into a pandas dataframe
print(data.shape)
data.drop(['id', 'date','sqft_basement','lat','long'], axis = 1, inplace = True)
print(data.shape)
for column in data:
    print(column,end=';')
print()

"""
categorial_cols = ['floors', 'view', 'condition', 'grade', 'bedrooms', 'bathrooms','zipcode']
for cc in categorial_cols:
    dummies = pd.get_dummies(data[cc], drop_first=False)
    dummies = dummies.add_prefix("{}#".format(cc))
    data.drop(cc, axis=1, inplace=True)
    data = data.join(dummies)
print(data.shape)
for column in data:
    print(column,end=';')
print()
"""

train_data, test_data = train_test_split(data, train_size = 0.8, random_state = 60)
train_data.astype(float).to_csv('kc_house_train_data.csv',sep=',',index=False, header=False)
test_data.astype(float).to_csv('kc_house_test_data.csv',sep=',',index=False, header=False)
