import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os


# Load the dataset
data = pd.read_csv('data/3d_linear_data.csv')

# Initialize parameters
m_x, m_y, b = 0.0, 0.0, 0.0
L = 0.01  # learning rate
epochs = 1000
frames_dir = 'frames_3d'
os.makedirs(frames_dir, exist_ok=True)

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

    #took from online the 3d graphing
    if epoch % 10 == 0 or epoch == epochs - 1:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data.x, data.y, data.z, color='blue', label='Data')

        # Create a meshgrid for the plane
        x_vals = np.linspace(0, 10, 10)
        y_vals = np.linspace(0, 10, 10)
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)
        z_grid = m_x * x_grid + m_y * y_grid + b

        # Plot the regression plane
        ax.plot_surface(x_grid, y_grid, z_grid, color='red', alpha=0.5)
        ax.set_title(f"Epoch {epoch}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elev=20, azim=135)  # optional: control camera angle

        plt.savefig(f"{frames_dir}/frame_{epoch:04d}.png")
        plt.close()


print(f"Trained parameters:\nm_x = {m_x:.4f}, m_y = {m_y:.4f}, b = {b:.4f}")
images = []
for filename in sorted(os.listdir(frames_dir)):
    if filename.endswith('.png'):
        images.append(imageio.imread(os.path.join(frames_dir, filename)))

imageio.mimsave('3d_gradient_descent.gif', images, duration=0.15)

print(" GIF saved as '3d_gradient_descent.gif'")