from neural_network import NeuralNetwork
from act_functions import sigmoid, sigmoid_derivative, \
    relu, relu_derivative, tanh, tanh_derivative
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

def regulize(Y):
    """Function scale Y values, output is in range [0, 1]
    
    return modified Y and used modifiers
    """

    min_value = 0
    max_value = 0
    for y in Y:
        if y < min_value:
            min_value = y
        if y > max_value:
            max_value = y
    Y = (Y - min_value) / (max_value - min_value)
    return Y, min_value, max_value

def show_learning(ticks_number, neural_network, train_X, train_y, \
         test_X, test_y, add_param=False):
    """Function train neural_network using train points
    
    and plot "learned" function every 100 steps
    Where:
    ticks_number -> after that number reset learning
    add_param -> if its true add extrpa polynomial X params
    """
    
    nn = neural_network
    orginal_y = train_y.copy()
    x_axis = train_X.copy()
    # Reshape X, y into matrix
    X = train_X.reshape(-1, 1)
    y = train_y.reshape(-1, 1)
    # Params used to scale ploted functions
    x_min = np.min(X)
    x_max = np.max(X)
    y_min = np.min(y)
    y_max = np.max(y)
    # Add polymonial params if add_param is True
    if add_param:
        X = np.c_[X, X**2]
    # Insert extra ones
    X = np.c_[X, np.ones(len(train_y))]
    # Scale y
    y, min_value_y, max_value_y = regulize(y)

    # Initiate animation params
    fig = plt.figure()
    ax = plt.axes(xlim=(x_min, x_max), ylim=(y_min, y_max))
    title = ax.text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")
    lines = []
    line, = ax.plot([], [], "bo" ,lw=3, label='"Learned" function')
    lines.append(line)
    line, = ax.plot([], [], "ro" ,lw=3, label="Orginal function")
    lines.append(line)

    def init():

        for line in lines:
            line.set_data([],[])
        return lines

    def animate(i, X=X, y=y):
        """side function used to prepare data for animation"""

        if i == 0:
            nn.insert_data(X.copy(), y.copy())
        # Every 50 step print error cost
        elif i % 50 == 0:
            print("Error cost:", nn.cost_function(orginal_y, \
                     min_value_y, max_value_y))

        for _ in range(100):
            nn.feedfoward()
            nn.backprop()
            y_axis = nn.output() * (max_value_y - min_value_y) + min_value_y
        x = x_axis
        y = y_axis
        lines[0].set_data(x, y)
        lines[1].set_data(test_X, test_y)
        title.set_text('"Learining" step: ' + str(i))

        return lines, title

    # Initiate animation
    anim = FuncAnimation(fig, animate, init_func=init,
                               frames=ticks_number, interval=50, blit=False)
    plt.legend(loc="lower left")
    plt.show()

def test_configuration(test_number, neural_network, train_X, train_y):
    """Function train neural_network and return error cost

    function used only for debugging
    Where test_number is a number of backprop call
    """
    
    np.random.seed(17)
    nn = neural_network
    orginal_y = train_y.copy()
    X = train_X.reshape(-1, 1)
    y = train_y.reshape(-1, 1)
    # Insert extra ones
    X = np.c_[X, np.ones(len(train_y))]
    y, min_value_y, max_value_y = regulize(y)
    nn.insert_data(X, y)
    for _ in range(test_number):
        nn.feedfoward()
        nn.backprop()
    return nn.cost_function(orginal_y, min_value_y, max_value_y)

    
if __name__ == "__main__":
    """Main module"""
    
    np.set_printoptions(precision=3, suppress=True)
    if len(sys.argv) != 5:
        print("Program takes exactly 4 arguments")
        print("Call can look like this: --type quad --ticks 100")
        quit()
    f_type = sys.argv[2]
    ticks_number = int(sys.argv[4])
    if f_type != "quad" and f_type != "sin":
        print("--type should be quad or sin")
        quit()
    elif f_type == "quad":
        eta = 0.02
        input_size = 1
        train_X = np.linspace(-5, 5, 101)
        train_y = train_X**2
        test_X = np.linspace(-5, 5, 26)
        test_y = test_X**2
        hl1_size = 15
        hl2_size = 15
        nn = NeuralNetwork([hl1_size, hl2_size, 1], [tanh, tanh, sigmoid], \
                [tanh_derivative, tanh_derivative, sigmoid_derivative], eta=eta)
        show_learning(ticks_number, nn, train_X, train_y, test_X, test_y, add_param=True)
    elif f_type == "sin":
        eta = 0.08
        input_size = 1
        train_X = np.linspace(0, 2, 161)
        train_y = np.sin((3*np.pi/2) * train_X)
        test_X = np.linspace(0, 2, 21)
        test_y = np.sin((3*np.pi/2) * test_X)
        hl1_size = 20
        hl2_size = 10
        nn = NeuralNetwork([hl1_size, hl2_size, 1], [tanh, sigmoid, sigmoid], \
                [tanh_derivative, sigmoid_derivative, sigmoid_derivative], eta=eta)
        show_learning(ticks_number, nn, train_X, train_y, test_X, test_y)