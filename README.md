# Car MPG Prediction Model

## Overview
This project implements a neural network model using PyTorch to predict the miles per gallon (MPG) of cars based on various features. The model is designed to learn from a dataset containing information about different car attributes.

## Model Description
The model consists of two layers:
1. A hidden layer with 4 neurons and a ReLU activation function.
2. An output layer with 1 neuron that predicts the MPG.

### Model Architecture
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 4)
        self.output = nn.Linear(4, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.hidden(x)
        output = self.relu(output)
        return self.output(output)
```

## Data
The dataset used for training and testing the model is loaded from a CSV file (`static/car_data.csv`). The relevant columns are:
- `combination_mpg`: The target variable (MPG).
- `fuel_type`: The type of fuel used by the car.
- `cylinders`: The number of cylinders in the car's engine.
- `year`: The year the car was manufactured.

### Data Processing Steps
1. Load data using the pandas package.
2. Drop any rows with missing values.
3. Encode categorical variables using `LabelEncoder`.
4. Convert the DataFrame to PyTorch tensors for model training.

## Training
The model is trained using Stochastic Gradient Descent (SGD) as the optimizer. The training process involves:
1. Defining the number of epochs (iterations).
2. Activating training mode.
3. Passing inputs through the neural network layers.
4. Calculating the loss using Mean Squared Error (MSE).
5. Backpropagating the loss to update the model weights.

### Training Code Snippet
```python
optimizer = SGD(model.parameters(), lr=0.001)
for i in range(epochs):
    model.train()
    output = model(train_X)
    loss = (train_y - output).pow(2).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Evaluation
After training, the model is evaluated on the test dataset. Predictions are made, and the results are visualized to compare actual vs. predicted MPG values.

### Evaluation Code Snippet
```python
with no_grad():
    test_predictions = model(test_X)
test_predictions = test_predictions.cpu().tolist()
test_actuals = test_y.cpu().tolist()
```

## Visualization
The results are plotted to show the comparison between actual and predicted MPG values.

```python
plt.plot(test_actuals, label='Actual Values', marker='o', linestyle='-', color='b')
plt.plot(test_predictions, label='Predicted Values', marker='x', linestyle='--', color='r')
```

## Conclusion
This project demonstrates the use of neural networks for regression tasks using PyTorch. The model can be further improved by experimenting with different architectures, hyperparameters, and additional features.
```

Feel free to modify any sections to better fit your project's specifics or to add any additional information you think is necessary!
