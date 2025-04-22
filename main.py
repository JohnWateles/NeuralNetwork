from neuro import NeuralNetwork, load_neural_network


def test1():
    layers = [2, 10, 2, 2]
    nn = NeuralNetwork(layers)

    nn.set_input_value([0.2, 0.8])
    nn.forward_propagation()
    nn.print()
    for i in range(100):
        nn.backward_propagation([1, 0], learning_rate=0.5)
        nn.forward_propagation()
    nn.print()
    output = nn.get_output_values()
    print(output)
    nn.save("testing")


def test2():
    nn = load_neural_network("testing")
    nn.set_input_value([0.2, 0.8])
    nn.forward_propagation()
    nn.print()


def main():
    pass


if __name__ == "__main__":
    main()
