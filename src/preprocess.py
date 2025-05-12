import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def preprocess_data():
    # Load training and testing data
    train_data = pd.read_csv("data/KDDTrain+.csv")
    test_data = pd.read_csv("data/KDDTest+.csv")

    # Clean up column names by stripping spaces
    train_data.columns = train_data.columns.str.strip()
    test_data.columns = test_data.columns.str.strip()

    # Print first few rows and check the last column for label
    print("First few rows of the train data:\n", train_data.head())
    print("Last column name:", train_data.columns[-1])

    # Assuming the last column is the label, extract it
    train_data['label'] = train_data[train_data.columns[-1]]
    test_data['label'] = test_data[test_data.columns[-1]]

    # Convert 'label' to binary (0 for 'normal', 1 for attacks)
    train_data['label'] = train_data['label'].apply(lambda x: 0 if x == 'normal' else 1)
    test_data['label'] = test_data['label'].apply(lambda x: 0 if x == 'normal' else 1)

    # Remove the label column for features
    X_train = train_data.drop(columns=['label'])
    y_train = train_data['label']
    X_test = test_data.drop(columns=['label'])
    y_test = test_data['label']

    # Encode categorical features (if any) using Label Encoding
    encoder = LabelEncoder()
    cat_cols = ['protocol_type', 'service', 'flag']
    for col in cat_cols:
        if col in X_train.columns:
            X_train[col] = encoder.fit_transform(X_train[col])
            X_test[col] = encoder.transform(X_test[col])

    # Convert all remaining non-numeric columns to numeric (if any)
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')

    # Now, only numerical columns should be scaled
    numeric_cols_train = X_train.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols_test = X_test.select_dtypes(include=['float64', 'int64']).columns

    # Ensure we use only common columns between train and test data
    common_numeric_cols = list(set(numeric_cols_train).intersection(set(numeric_cols_test)))

    # Align test set columns with train set
    X_test = X_test[common_numeric_cols]
    X_train = X_train[common_numeric_cols]

    # Create the MinMaxScaler instance and scale the data
    scaler = MinMaxScaler()
    X_train[common_numeric_cols] = scaler.fit_transform(X_train[common_numeric_cols])
    X_test[common_numeric_cols] = scaler.transform(X_test[common_numeric_cols])

    return X_train, y_train, X_test, y_test
