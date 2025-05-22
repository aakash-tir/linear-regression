import pandas as pd

# Load the dataset
data = pd.read_csv('data/3d_linear_data.csv')

# Initialize parameters
m_x, m_y, b = 0.0, 0.0, 0.0
L = 0.01  # learning rate
epochs = 1000


def gradient_descent(m_x_now, m_y_now, b_now, data, L):
    m_x_grad, m_y_grad, b_grad = 0, 0, 0
    n = len(data)

    for i in range(n):
        x_i = data.iloc[i].x
        y_i = data.iloc[i].y
        z_i = data.iloc[i].z
        prediction = m_x_now * x_i + m_y_now * y_i + b_now
        error = z_i - prediction

        # Partial derivatives based on dx and dy
        m_x_grad += (-2/n) * x_i * error
        m_y_grad += (-2/n) * y_i * error
        b_grad   += (-2/n) * error

    # Update parameters
    m_x_new = m_x_now - L * m_x_grad
    m_y_new = m_y_now - L * m_y_grad
    b_new   = b_now - L * b_grad

    return m_x_new, m_y_new, b_new

# Run gradient descent
for epoch in range(epochs):
    m_x, m_y, b = gradient_descent(m_x, m_y, b, data, L)

print(f"Trained parameters:\nm_x = {m_x:.4f}, m_y = {m_y:.4f}, b = {b:.4f}")
