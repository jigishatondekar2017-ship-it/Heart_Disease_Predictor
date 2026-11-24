import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import(
accuracy_score,
classification_report,
confusion_matrix
)
#Load Dataset
df=pd.read_csv('heart_disease.csv')
print('Dataset Shape:',df.shape)
print(df.head())
print(df.isnull().sum())

#Remove missing values
df=df.dropna()

#Separate features and target
X=df.drop('target',axis=1)
y=df['target']

#Split dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

#Define ML algorithms
models={
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100,random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf')
}
best_score=-1
best_model=None
best_name=""

#Train and compare model
for name,algorithm in models.items():
    pipeline=Pipeline([
        ('scaler',StandardScaler()),
        ('model',algorithm)
    ])
    scores = cross_val_score(pipeline,
                             X_train,
                             y_train,
                             cv=5,
                             scoring='accuracy')
    print(name,'CV Accuracy:',scores.mean())
    if scores.mean()>best_score:
        best_score=scores.mean()
        best_model=pipeline
        best_name=name

#Train Selected Model
best_model.fit(X_train,y_train)

#Test Model
y_pred=best_model.predict(X_test)

print('Best Model:',best_name)
print('Test Accuracy:',accuracy_score(y_test,y_pred))
print('Classification Report:')
print(classification_report(y_test,y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test,y_pred))

#Save model
joblib.dump(best_model,'heart_model.pkl')

#Save feature order
joblib.dump(list(X.columns),'features.pkl')


print('Model Saved Successfully................')