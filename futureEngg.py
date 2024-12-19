import pandas as pd
df=pd.read_csv('modifiedTiatanic2.csv')
#Create Family Size:

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

#Create IsAlone:
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

#Bin Age into Groups:
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 80], labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])



#Bin Fare into Groups
df['FareGroup'] = pd.qcut(df['Fare'], 4, labels=['Low', 'Medium', 'High', 'Very High'])




print(df.head(10))

df=df.to_csv('modifiedTiatanic3.csv',index=False)

