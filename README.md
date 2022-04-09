<p align="left"><img src="https://cdn-images-1.medium.com/max/184/1*2GDcaeYIx_bQAZLxWM4PsQ@2x.png"></p>

# __Diamonds - Machine Learning__ :gem:

Ironhack Madrid - Data Analytics Part Time - Noviembre 2021 - Proyecto Módulo 3

I have generated this machine learning model to predict the price of diamonds by participating in a kaggle competition.

<p align="center"><img src="https://media.giphy.com/media/l41JHqxTq1RRTZuso/giphy.gif"></p>

---

## **Data:** :books:

There are two data sets. The first contains the prices and other attributes of the diamonds, and was used to train the model. The other contains only attributes, which is used to predict the price of those diamonds. 

It is necessary to import the diamonds_train.db and diamonds_test.csv, and make several queries and convert it into a dataframe.

### **Dataset Columns** :

**price** : price in US dollars 

**carat** : weight of the diamond 

**cut** : quality of the cut 

**color** : diamond colour

**clarity** : clear the diamond

**x** : length in mm 

**y** : width in mm 

**z** : depth in mm 

**depth** : total depth percentage = z / mean( x, y ) = 2 * z / ( x + y ) 

**table** : width of top of diamond relative to widest point 

---
## **Machine Learning** :brain:
We are dealing with a supervised and regression model, so I have decided to use the algorithm Random forest. This information helped me to decide:

"Random forest is a technique used in prediction modeling and behavioral analysis and is based on decision trees. It contains many decision trees that represent a different instance of the classification of the data input in the random forest. The random forest technique considers the instances individually, taking as the selected prediction the one with the most votes.

Random forests feature variable importance estimates, i.e., neural networks. They also offer a superior method for working with missing data. Missing values are replaced by the variable that appears the most at a particular node. Among all available classification methods, random forests provide the highest accuracy.

The random forest technique can also handle large data with numerous variables numbering in the thousands. It can automatically balance data sets when one class is less frequent than other classes in the data. The method also handles variables quickly, making it suitable for complicated tasks."

 Corporate finance institute
  https://corporatefinanceinstitute.com/resources/knowledge/other/random-forest/

The way to measure the prediction is by calculating the RMSE on my notebook. For this I split the dataset using "train_test_split" from the sklearn library. I used 80% to train the model, and the remaining 20% to compare the prediction. But as it is a kaggle competition, the result was trained with the whole dataset.

To improve the model, I applied sklearn's OrdinalEncoder to the categories. I decided to leave the outliers to adjust it to reality, and the result in the competition improved. 
Finally I scaled the data so that the result of the model improved. 
<p align="center"><img src="https://media.giphy.com/media/lSuTC8GwJUsZpMUThd/giphy.gif"></p>


---

## **Code fragments:** :rocket:

### **SQL** :
```
SELECT 
            propertis.index_id,
            clarity.clarity,
            color.color,
            cut.cut,
            dimensions.depth,
            dimensions.'table',
            dimensions.x,
            dimensions.y,
            dimensions.z,
            transactional.price,
            transactional.carat,
            city.city
        FROM diamonds_properties as propertis
        LEFT JOIN diamonds_clarity as clarity ON propertis.clarity_id = clarity.clarity_id
        LEFT JOIN diamonds_color as color ON propertis.color_id = color.color_id
        LEFT JOIN diamonds_cut as cut ON propertis.cut_id = cut.cut_id
        LEFT JOIN diamonds_dimensions as dimensions ON propertis.index_id = dimensions.index_id
        LEFT JOIN diamonds_transactional as transactional ON propertis.index_id = transactional.index_id
        LEFT JOIN diamonds_city as city ON transactional.city_id = city.city_id
```

### **OrdinalEncoder** :
```
encoder = OrdinalEncoder(categories=[['J', 'I', 'H', 'G', 'F', 'E', 'D']])

encoder.fit(train_df[["color"]])
train_df["color-encoded"] = encoder.transform(train_df[["color"]])
train_df.drop("color", axis = 1 , inplace = True)

encoder = OrdinalEncoder(categories=[['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']])
encoder.fit(train_df[["cut"]])
train_df["cut-encoded"] = encoder.transform(train_df[["cut"]])
train_df.drop("cut", axis = 1 , inplace = True)

encoder = OrdinalEncoder(categories=[['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']])
encoder.fit(train_df[["clarity"]])
train_df["clarity-encoded"] = encoder.transform(train_df[["clarity"]])
train_df.drop("clarity", axis = 1 , inplace = True)
```

### **Prepare the dataset.** :
```
target='price'
cat_features = ['cut-encoded','clarity-encoded','color-encoded']
num_features = ['depth','table','x','y','z','carat']
for cat_feat in cat_features:
    train_df[cat_feat]=train_df[cat_feat].astype('category')
    test_df[cat_feat]=test_df[cat_feat].astype('category')


cat_df = train_df[cat_features]
num_df = train_df.loc[:,num_features]
df_train = pd.concat([cat_df, num_df], axis=1)


cat_df = test_df[cat_features]
num_df = test_df.loc[:,num_features]
df_test = pd.concat([cat_df, num_df], axis=1)
    
features = list(cat_df.columns) + list(num_df.columns)
```

### **Scaler** :
```
scaler = StandardScaler()
X=scaler.fit_transform(df_train.loc[:,features].values)
y=train_df[target]
```

## **References:** :hammer_and_wrench:

- [SQLAlchemy](https://docs.sqlalchemy.org/en/14/core/engines.html)

- [Pandas](https://pandas.pydata.org/docs/reference/index.html)

- [Seaborn](https://seaborn.pydata.org/)

- [sklearn](https://scikit-learn.org/stable/user_guide.html)
