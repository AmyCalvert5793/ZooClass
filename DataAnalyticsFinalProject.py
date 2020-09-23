#Amy Calvert
#Data Analytics
#Resources Used - Pandas cheat sheet, 




 

# data processing
import pandas as pd


# data visualization
from matplotlib import pyplot as plt
from matplotlib import style
from pandas.plotting import andrews_curves
from pandas.plotting import radviz
from pandas.plotting import parallel_coordinates

# Algorithms
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")


#import data
zoo_df = pd.read_csv("zoo.csv")
class_df = pd.read_csv("class.csv")


#drop animal name for visualization
new_zoo = zoo_df.iloc[:,1:]

#drop hair 
new_zoo2 = zoo_df.iloc[:,1:]



#Data Visualizations

#shows sum of characteristics per category
#zoo_df.groupby('class_type').sum().plot(kind = 'bar');

#shows total count of each class
#sns.countplot(zoo_df['class_type'],label="Count")

#radviz
#plt.figure()
#radviz(new_zoo, 'class_type')

#andrew_curves
#andrews_curves(data, 'Name', colormap='winter')

#zoo_df.plot.hexbin(x='class_type', y='hair', gridsize=25)


#correlation heat matrix
#corr = zoo_df.iloc[:,1:-1].corr()
#colormap = sns.diverging_palette(220, 10, as_cmap = True)
#plt.figure(figsize=(14,14))
#sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 12},
#            cmap = colormap, linewidths=0.1, linecolor='white')
#plt.title('Correlation of ZOO Features', y=1.05, size=15)



X = zoo_df.iloc[:,:-1]

y = zoo_df.iloc[:,-1:]

#split data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=14, stratify=y)


#save animal names for later, drop for now 
keep_train_name = train_X['animal_name']
keep_test_name = test_X['animal_name']

train_X = train_X.iloc[:,1:]
test_X = test_X.iloc[:,1:]


#Naive Bayes Calssifier
NBC=GaussianNB().fit(train_X,train_y)
s=NBC.predict(test_X)
print(accuracy_score(test_y,s))
GNB_ACC=NBC.score(test_X,test_y)

#plt.title('Gaussian Naive Bayes')
#plt.scatter(test_y, s)
#plt.xlabel('Real Values')
#plt.ylabel('Predictions')


#K Nearest Neighbors
KNN=KNeighborsClassifier(n_neighbors=3).fit(train_X,train_y)
s2=KNN.predict(test_X)
print(accuracy_score(test_y,s2))
KNN_ACC=KNN.score(test_X,test_y)

#plt.title('K Nearest Neighbors')
#plt.scatter(test_y, s2)
#plt.xlabel('Real Values')
#plt.ylabel('Predictions')


#Support Vector Machine
SVM=svm.SVC(gamma=1,C=1).fit(train_X,train_y)
s3=SVM.predict(test_X)
print(accuracy_score(test_y,s3))
SVM_ACC=SVM.score(test_X,test_y)

#plt.title('Support Vector Machine')
#plt.scatter(test_y, s3)
#plt.xlabel('Real Values')
#plt.ylabel('Predictions')


#Logistic Regression
LRC = LogisticRegression()
LRC.fit(train_X, train_y)
s4 = LRC.predict(test_X)
print(accuracy_score(test_y,s4))
LRC_ACC=LRC.score(test_X,test_y)

plt.title('Logistic Regression')
plt.scatter(test_y, s4)
plt.xlabel('Real Values')
plt.ylabel('Predictions')
plt.show()
      
#Y=[GNB_ACC*100,KNN_ACC*100,SVM_ACC*100,LRC_ACC*100]
#X=['GNB','KNN','SVM','LRC']

#plt.bar(X,Y)

