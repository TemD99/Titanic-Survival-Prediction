import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

if __name__ == '__main__':
    seed = 1234123

    current_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_directory)

    data_dir = os.path.join(project_root, 'data')
    out_dir = os.path.join(project_root, 'output')
    file_name = 'titanic_passengers.csv'
    path = os.path.join(data_dir, file_name)

    df = pd.read_csv(path, encoding='ISO-8859-1')
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    df.info()
    labelencoder = LabelEncoder()
    df['Sex'] = labelencoder.fit_transform(df['Sex'])
   
    features = df.drop('Survived', axis=1)
    target = df['Survived']

    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(features)

    # Impute Age
    imputer = KNNImputer(n_neighbors=5)
    features_standardized_imputed = imputer.fit_transform(features_standardized)

    # Recreating data frame
    df_standardized_imputed = pd.DataFrame(features_standardized_imputed, columns=features.columns)
    df_standardized_imputed['Survived'] = target
    df_standardized_imputed.info()
    
   # First Heatmap
    corr = df_standardized_imputed.corr()
    sns.heatmap(corr, annot=True)
    plt.savefig(os.path.join(out_dir, 'heatmap_all_corr.png'))  
    plt.show()

    # Second Heatmap
    corr = corr.abs()
    sns.heatmap(corr, vmin=0.9, vmax=1, annot=True)
    plt.savefig(os.path.join(out_dir, 'heatmap_high_corr.png'))  
    plt.show()
    
   
    X = df_standardized_imputed.drop('Survived', axis=1)
    y = df_standardized_imputed['Survived']

# Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Create a Logistic Regression model
    logreg = LogisticRegression()

# Fit the model to the training data
    logreg.fit(X_train, y_train)

# Make predictions on the test set
    y_pred = logreg.predict(X_test)
    train_score = logreg.score(X_train, y_train)
    test_score = logreg.score(X_test, y_test)
# Evaluate the model
    print(train_score)
    print(test_score)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    