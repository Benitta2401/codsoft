
#####convert categorical variables into numerical values
import pandas as pd


from sklearn.preprocessing import LabelEncoder

df1=pd.read_csv('modifiedTitanic.csv')

label_encoder = LabelEncoder()
df1['Sex'] = label_encoder.fit_transform(df1['Sex'])  # Male: 1, Female: 0
df1['Pclass'] = label_encoder.fit_transform(df1['Pclass'])  # 1st, 2nd, 3rd as ordinal
#one hot encoding
df1['Embarked'] = df1['Embarked'].fillna(df1['Embarked'].mode()[0])

# print(df1.head(200))

df1=df1.to_csv('modifiedTiatanic2.csv',index=False)
# print(df1.head(200))