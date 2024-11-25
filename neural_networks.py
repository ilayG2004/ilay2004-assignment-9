import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)


#Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def relu(x):
    return np.maximum(0, x)
def tanh(x):
    return np.tanh(x)

'''
Input layer: Dimensionality of 2. [x1, x2] 
Hidden layer: 1 hidden layer with 3 neurons.
Output layer: Single output for binary classification.
'''
# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0) 
        self.lr = lr
        self.activation_fn = activation

        self.weights_input_hidden = np.random.randn(input_dim, hidden_dim) * 0.1    #Intialize weight 1
        self.bias_hidden = np.zeros((1, hidden_dim))                                #Intitialize bias 1
        self.weights_output_hidden = np.random.randn(hidden_dim, output_dim) * 0.1  #Intitialize weight 2
        self.bias_output = np.zeros((1, output_dim))                                #Intialize bias 2

        self.hlayer_activations = None
        self.gradients = {}                                                            

    def forward(self, X):
        z_hidden = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        #Pass activation function on hidden layers
        if self.activation_fn == 'tanh':
            self.hlayer_activations = tanh(z_hidden)
        elif self.activation_fn == 'relu':
            self.hlayer_activations = relu(z_hidden)
        elif self.activation_fn == 'sigmoid':
            self.hlayer_activations = sigmoid(z_hidden)

        z_output = np.dot(self.hlayer_activations, self.weights_output_hidden) + self.bias_output
        out =  np.tanh(z_output)
        return out

    def backward(self, X, y):
        #Cross entropy loss
        output_error = self.forward(X) - y
        delta_output = output_error * (1 - self.forward(X)**2)

        #Gradient descent
        grad_weights_output_hidden = np.dot(self.hlayer_activations.T, delta_output)
        grad_bias_output = np.sum(delta_output, axis=0, keepdims=True)

        if self.activation_fn == 'sigmoid':
            delta_hidden = np.dot(delta_output, self.weights_output_hidden.T) * (self.hlayer_activations * (1 - self.hlayer_activations))
        elif self.activation_fn == 'relu':
            delta_hidden = np.dot(delta_output, self.weights_output_hidden.T) * (self.hlayer_activations > 0)
        elif self.activation_fn == 'tanh':
            delta_hidden = np.dot(delta_output, self.weights_output_hidden.T) * (1 - self.hlayer_activations**2)

        grad_weights_input_hidden = np.dot(X.T, delta_hidden)
        grad_bias_hidden = np.sum(delta_hidden, axis=0, keepdims=True)

        self.weights_input_hidden -= self.lr * grad_weights_input_hidden
        self.weights_output_hidden -= self.lr * grad_weights_output_hidden

        self.bias_hidden -= self.lr * grad_bias_hidden
        self.bias_output -= self.lr * grad_bias_output

        self.gradients = {
            'input_hidden': grad_weights_input_hidden,
            'output_hidden': grad_weights_output_hidden
        }

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # TODO: Plot hidden features
    hidden_features = mlp.hlayer_activations
    ax_hidden.scatter(
        hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], 
        c=y.ravel(), cmap='bwr', alpha=0.7
    )
    ax_hidden.set_title(f"Hidden Space at Step {frame * 10}")
    ax_hidden.set_xlabel("Hidden Dimension 1")
    ax_hidden.set_ylabel("Hidden Dimension 2")
    ax_hidden.set_zlabel("Hidden Dimension 3")

    x_vals = np.linspace(-1.5, 1.5, 50)
    y_vals = np.linspace(-1.5, 1.5, 50)
    xx, yy = np.meshgrid(x_vals, y_vals)
    z_vals = -(mlp.weights_output_hidden[0, 0] * xx +
                mlp.weights_output_hidden[1, 0] * yy +
                mlp.bias_output[0, 0]) / (mlp.weights_output_hidden[2, 0] + 1e-5)
    ax_hidden.plot_surface(xx, yy, z_vals, alpha=0.3, color='tan')

    # TODO: Distorted input space transformed by the hidden layer
    # TODO: Plot input layer decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = mlp.forward(grid).reshape(xx.shape)

    ax_input.contour(xx, yy, predictions, levels=[0], colors='black', linewidths=1.5)
    ax_input.contourf(xx, yy, predictions, levels=[-1, 0, 1], colors=['red', 'blue'], alpha=0.5)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k', s=20)
    ax_input.set_title(f"Input Space at Step {frame * 10}")
    ax_input.set_xlabel("Feature 1")
    ax_input.set_ylabel("Feature 2")

    # TODO: Visualize features and gradients as circles and edges 
    # The edge thickness visually represents the magnitude of the gradient
    ax_gradient.set_title(f"Gradients at Step {frame * 10}")
    ax_gradient.set_xlim(0, 1)
    ax_gradient.set_ylim(0, 1)
    ax_gradient.axis('off')

    nodes = {
        'x1': (0.2, 0.8), 'x2': (0.2, 0.6),
        'h1': (0.5, 0.9), 'h2': (0.5, 0.7), 'h3': (0.5, 0.5),
        'y': (0.8, 0.7)
    }

    for name, (x, y) in nodes.items():
        ax_gradient.add_patch(Circle((x, y), 0.03, color='blue'))
        ax_gradient.text(x, y, name, color='white', ha='center', va='center')

    edges = [
        ('x1', 'h1', mlp.gradients['input_hidden'][0, 0]),
        ('x1', 'h2', mlp.gradients['input_hidden'][0, 1]),
        ('x1', 'h3', mlp.gradients['input_hidden'][0, 2]),
        ('x2', 'h1', mlp.gradients['input_hidden'][1, 0]),
        ('x2', 'h2', mlp.gradients['input_hidden'][1, 1]),
        ('x2', 'h3', mlp.gradients['input_hidden'][1, 2]),
        ('h1', 'y', mlp.gradients['output_hidden'][0, 0]),
        ('h2', 'y', mlp.gradients['output_hidden'][1, 0]),
        ('h3', 'y', mlp.gradients['output_hidden'][2, 0]),
    ]

    for start, end, grad in edges:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        linewidth = min(3, max(0.5, abs(grad) * 5)) 
        ax_gradient.plot([x1, x2], [y1, y2], 'm-', linewidth=linewidth)


def generate_data(n_samples=100):
    #Random dataset.
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)