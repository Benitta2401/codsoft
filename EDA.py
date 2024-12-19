import pandas as pd
df=pd.read_csv('modifiedTiatanic2.csv')
# print(df.head(20))

#step1:
#summary Statistics
# print(df.describe())
# df.info()

#step2:
# Visualizing Survival Distribution
#overall survival rate

import seaborn as sns
import matplotlib.pyplot as plt

# sns.countplot(x='Survived', data=df)
# plt.title('Survival Distribution')
# plt.show()


#step3
#Analyzing Categorical Variables
#Survival by Gender:
# sns.barplot(x='Sex', y='Survived', data=df)
# plt.title('Survival by Gender')
# plt.show()


#survival by passengerclass
# sns.barplot(x='Pclass', y='Survived', data=df)
# plt.title('Survival by Passenger Class')
# plt.show()

#Survival by Embarked:
# sns.barplot(x='Embarked', y='Survived', data=df)
# plt.title('Survival by Embarked Port')
# plt.show()

#step4
#Analyzing Numeric Variables
#Age:
# sns.histplot(data=df, x='Age', hue='Survived', kde=True, bins=30)
# plt.title('Age Distribution by Survival')
# plt.show()

#Fare:
# sns.boxplot(x='Survived', y='Fare', data=df)
# plt.title('Fare Distribution by Survival')
# plt.show()

# df = df.drop(['Name','Ticket'],axis=1)
# print(df.head(20))

# step5
# correlation matrix

# corr_matrix = df.corr()
# corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()#Excluding non numeric values
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()

#step6
#Relationships Between Multiple Variables
#Survival by Gender and Class:
# sns.catplot(x='Pclass', hue='Sex', col='Survived', kind='count', data=df)
# plt.show()

#Survival by Age and Class:
# sns.boxplot(x='Pclass', y='Age', hue='Survived', data=df)
# plt.title('Age and Class by Survival')
# plt.show()

