from src.preprocess import preprocess_data
from src.model import train_model
from src.utils import evaluate_model

# Step 1: Load and preprocess data
X_train, y_train, X_test, y_test = preprocess_data()

# Step 2: Train the model
model = train_model(X_train, y_train)

# Step 3: Evaluate the model
evaluate_model(model, X_test, y_test)
