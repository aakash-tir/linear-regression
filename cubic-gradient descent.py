import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


#original function
def f(x):
    return x**3 - 6*x**2 + 4*x + 12

def f_prime(x):
    return 3*x**2 - 12*x + 4


#function for reusability
def gradient_descent_step(x, learning_rate ,x_min, x_max):
    new_x = x - learning_rate * f_prime(x)
    if new_x < x_min:
        return x_min, True
    elif new_x > x_max:
        return x_max, True
    else:
        return new_x, False



# Parameters
L = 0.01
epochs = 100
x_min, x_max = -2, 6
frames_dir = "cubic_frames"
os.makedirs(frames_dir, exist_ok=True)

#start points
x1 = -1.5   
x2 = 5.5    

x1_path = [x1]
x2_path = [x2]
f1_path = [f(x1)]
f2_path = [f(x2)]

#freeze if bounds hit
frozen1 = False
frozen2 = False

for epoch in range(epochs):
    if not frozen1:
        x1, frozen1 = gradient_descent_step(x1, L, x_min, x_max)
    if not frozen2:
        x2, frozen2 = gradient_descent_step(x2, L, x_min, x_max)

    x1_path.append(x1)
    x2_path.append(x2)
    f1_path.append(f(x1))
    f2_path.append(f(x2))


    # ploting graph

    if epoch % 2 == 0 or epoch == epochs - 1:
        x_plot = np.linspace(-2, 6, 400)
        y_plot = f(x_plot)

        plt.figure()
        plt.plot(x_plot, y_plot, label='f(x)', color='blue')

        plt.scatter(x1_path, f1_path, color='red', s=15, label='Path 1 (start: -1.5)')
        plt.scatter(x2_path, f2_path, color='green', s=15, label='Path 2 (start: 5.5)')

        plt.title(f'Epoch {epoch}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{frames_dir}/frame_{epoch:03d}.png")
        plt.close()

#GIF
images = []
for filename in sorted(os.listdir(frames_dir)):
    if filename.endswith(".png"):
        images.append(imageio.imread(os.path.join(frames_dir, filename)))

imageio.mimsave("cubic_descent.gif", images, duration=0.2)
print(" Dual-path gradient descent GIF saved as 'cubic_descent.gif'")
