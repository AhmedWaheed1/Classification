# made by eng/Ahmed Waheed
import pandas
trainData = pandas.read_csv("training.csv"   , sep = ";")
testData  = pandas.read_csv("validation.csv" , sep = ";")

from sklearn.impute import SimpleImputer
imputer   = SimpleImputer(strategy='most_frequent')
trainData = pandas.DataFrame(imputer.fit_transform(trainData))
testData  = pandas.DataFrame(imputer.    transform(testData ))

train     = pandas.DataFrame(imputer.fit_transform(trainData))#copy
test      = pandas.DataFrame(imputer.    transform(testData ))#copy

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
columString = [0,1,2,3,4,5,6,7,8,9,11,12,16]
for i in columString:
    encoder.fit( pandas.concat( [ trainData.iloc[:,i] , testData.iloc[:,i] ] ) )
    trainData.iloc[:,i] = encoder.transform(trainData.iloc[:,i])
    testData .iloc[:,i] = encoder.transform(testData .iloc[:,i])

X_train = trainData.drop(18,1)#Remove label Column
Y_train = trainData[18]
X_test  = testData.drop(18,1)
Y_test  = testData[18]

from sklearn.naive_bayes import GaussianNB
model = GaussianNB(var_smoothing = 1e-12)
model.fit(X_train , Y_train)
predict = model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("accuracy = " + str( accuracy_score (Y_test , predict)) ) # 80 %
print( confusion_matrix                   (Y_test , predict))
print( classification_report              (Y_test , predict))

x=input('''enter a instance to predict (delimiter= ;) as this format
a;17,92;5,4e-05;u;g;c;v;1,75;f;t;1;t;g;80;5;8.00E+05;t;0 \n''')
x=pandas.DataFrame( x.split(';') ).T
for i in columString:
    encoder.fit( pandas.concat( [ train.iloc[:,i] , test.iloc[:,i] ] ) )
    x.iloc[:,i]= encoder.transform(x.iloc[:,i])
print('label is ' + str(model.predict(x)))