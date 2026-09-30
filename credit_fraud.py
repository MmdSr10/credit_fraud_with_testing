import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_inspect_data(file_path:str) -> pd.DataFrame :

    """Load dataset and display basic statistics."""
    df=pd.read_csv(file_path)

    print(f"Dataset Shape: {df.shape}")
    print("\nClass Distribution:")
    print(df["Class"].value_counts())

    return df

def split_and_scale_data(df:pd.DataFrame):
    """Split raw data to train/test and scale 'Time' and 'Amount' features."""

    X=df.drop(columns="Class")
    y=df['Class']

    #Splitting Data

    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=28,test_size=0.2,stratify=y)

    #Standardization

    scaler=StandardScaler()

    cols_to_scale=['Time','Amount']

    X_train[cols_to_scale]=scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale]=scaler.transform(X_test[cols_to_scale])

    return X_train, X_test, y_train, y_test

def apply_undersampling(X_train: pd.DataFrame, y_train: pd.Series):
  """Balance training dataset only using undersampling."""

  train_data = pd.concat([X_train, y_train], axis=1)

  legit = train_data[train_data.Class == 0]
  fraud = train_data[train_data.Class == 1]


  legit_sample = legit.sample(n=len(fraud), random_state=42)
  balanced_train = pd.concat([legit_sample, fraud], axis=0).sample(
      frac=1, random_state=42
  )

  X_train_balanced = balanced_train.drop(columns="Class")
  y_train_balanced = balanced_train["Class"]

  return X_train_balanced, y_train_balanced

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
  """Train Logistic Regression model."""
  model = LogisticRegression(max_iter=1000, random_state=42)
  model.fit(X_train, y_train)
  return model


def evaluate_model(
    model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series
):
  """Evaluate model performance on the imbalanced test set."""
  preds = model.predict(X_test)
  preds_proba = model.predict_proba(X_test)[:, 1]

  print("\n" + "=" * 50)
  print("EVALUATION ON ORIGINAL TEST SET")
  print("=" * 50)

  print("\nConfusion Matrix:")
  print(confusion_matrix(y_test, preds))

  print("\nClassification Report:")
  print(classification_report(y_test, preds, digits=4))

  print(f"ROC-AUC Score: {roc_auc_score(y_test, preds_proba):.4f}")


def main():
  data_path = "creditcard.csv"

  df = load_and_inspect_data(data_path)
  X_train, X_test, y_train, y_test = split_and_scale_data(df)
  X_train_balanced, y_train_balanced = apply_undersampling(X_train, y_train)

  model = train_model(X_train_balanced, y_train_balanced)
  evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
  main()