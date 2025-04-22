from sklearn.datasets import fetch_openml
import numpy as np
from tqdm import tqdm

from neuro import NeuralNetwork, load_neural_network


def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]


def check(x_test, y_test, nn):
    correct = 0
    for x, y_true in zip(x_test, y_test):
        nn.set_input_value(x.tolist())
        nn.forward_propagation()

        output = np.array(nn.get_output_values())
        predicted = np.argmax(output)

        if predicted == y_true:
            correct += 1

    accuracy = correct / min(len(x_test), len(y_test))
    return accuracy


def training():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    x, y = mnist["data"], mnist["target"]

    x = x / 255.0
    y = y.astype(np.uint8)
    x_train = x[:10000]
    y_train = y[:10000]

    y_train_onehot = one_hot(y_train)

    layers = [28 * 28, 128, 10]
    nn = NeuralNetwork(layers)

    # nn = load_neural_network("MNIST")

    print(f"Input layer size: {nn.len_input_layer}")
    print(f"Output layer size: {nn.len_output_layer}")

    epochs = 3
    learning_rate = 0.5
    batch_size = 32

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train_onehot[i:i + batch_size]

            batch_loss = 0.0
            for x, y_true in zip(x_batch, y_batch):
                nn.set_input_value(x)
                nn.forward_propagation()
                nn.backward_propagation(y_true, learning_rate)
                output = nn.get_output_values()
                batch_loss += np.mean((output - y_true) ** 2)

            epoch_loss += batch_loss / batch_size
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(x_train) * batch_size:.10f}")
    nn.save("MNIST")


def main():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    x, y = mnist["data"], mnist["target"]

    x = x / 255.0
    y = y.astype(np.uint8)
    x_test = x[60000:]
    y_test = y[60000:]

    nn = load_neural_network("MNIST")
    value = check(x_test, y_test, nn)
    print(value)


if __name__ == "__main__":
    training()
    main()
