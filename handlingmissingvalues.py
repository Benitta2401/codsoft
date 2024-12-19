import pandas as pd
df=pd.read_csv('workingTitanic.csv')
df.info()
print(df.shape)
# print(df.head(5))
# print(df.isnull().sum())
# missing_percentage=df.isnull().mean()*100
# print(missing_percentage)
# print(df.head())
# df['Age'] = df['Age'].fillna(df.groupby(['Pclass', 'Sex'])['Age'].transform('median'))
# df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# df.to_csv("modifiedTitanic.csv",index=False)
# # df1=pd.read_csv('modifiedTitanic.csv',index=False)
# # print(df1.isnull().sum())




