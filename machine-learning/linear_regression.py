import numpy as np

def train_linear_regression(X, y, learning_rate=0.01, iterations=1000):
    """
    Trains a linear regression model using Gradient Descent.
    
    Parameters:
    X : Input data (features)
    y : Target values (labels)
    learning_rate : How big of a step we take down the gradient
    iterations : How many times we update the parameters
    """
    
    # 1. Initialize parameters
    # a = slope, b = intercept. We start at 0.
    a = 0.0
    b = 0.0
    n = len(y) # Number of data points
    
    print(f"Starting training on {n} data points...")

    for i in range(iterations):
        # 2. Make a prediction (y = mx + b)
        y_prediction = a * X + b
        
        # 3. Calculate the Error (Cost)
        # We use Mean Squared Error (MSE)
        cost = (1/n) * sum((y - y_prediction)**2)
        
        # 4. Calculate Gradients (the direction of steepest ascent)
        # These are the partial derivatives of the cost function
        da = (-2/n) * sum(X * (y - y_prediction))
        db = (-2/n) * sum(y - y_prediction)
        
        # 5. Update m and b
        # We subtract the gradient because we want to go DOWN the hill
        a = a - (learning_rate * da)
        b = b - (learning_rate * db)
        
        # Print progress every 100 steps
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost:.4f}, a {a:.4f}, b {b:.4f}")
            
    return a, b

# --- Let's test it with some sample data ---

# Example: X is hours studied, y is test score
# We expect a positive correlation
X_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2, 4, 5, 4, 5])

# Train the model
final_a, final_b = train_linear_regression(X_data, y_data, learning_rate=0.01, iterations=1000)

print("\n--- Training Complete ---")
print(f"Final Equation: y = {final_a:.2f}x + {final_b:.2f}")

# 6. Make a new prediction
new_x = 6
prediction = final_a * new_x + final_b
print(f"Prediction for x={new_x}: {prediction:.2f}")
