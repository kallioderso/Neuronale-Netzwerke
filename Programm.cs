using System;
using System.Collections.Generic;

// Netzwerk mit 2 Eingaben, 1 Hidden Layer mit 3 Neuronen, und 1 Output-Neuron
NeuralNetwork net = new NeuralNetwork(new int[] { 4, 8, 12, 16, 16});
double learningRate = 0.1;

for (int epoch = 0; epoch < 100000; epoch++)
{
    net.Train(new double[] { 0, 0, 0, 0 }, new double[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, learningRate);
    net.Train(new double[] { 0, 0, 0, 1 }, new double[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, learningRate);
    net.Train(new double[] { 0, 0, 1, 0 }, new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, learningRate);
    net.Train(new double[] { 0, 0, 1, 1 }, new double[] { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, learningRate);
    net.Train(new double[] { 0, 1, 0, 0 }, new double[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, learningRate);
    net.Train(new double[] { 0, 1, 0, 1 }, new double[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, learningRate);
    net.Train(new double[] { 0, 1, 1, 0 }, new double[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, learningRate);
    net.Train(new double[] { 0, 1, 1, 1 }, new double[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 }, learningRate);
    net.Train(new double[] { 1, 0, 0, 0 }, new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 }, learningRate);
    net.Train(new double[] { 1, 0, 0, 1 }, new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 }, learningRate);
    net.Train(new double[] { 1, 0, 1, 0 }, new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 }, learningRate);
    net.Train(new double[] { 1, 0, 1, 1 }, new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 }, learningRate);
    net.Train(new double[] { 1, 1, 0, 0 }, new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 }, learningRate);
    net.Train(new double[] { 1, 1, 0, 1 }, new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 }, learningRate);
    net.Train(new double[] { 1, 1, 1, 0 }, new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 }, learningRate);
    net.Train(new double[] { 1, 1, 1, 1 }, new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, learningRate);
}

while (true)
{
    string input = Console.ReadLine();
    string[] parts = input.Split(' ');
    double[,] inputs = new double[parts.Length, net.Layers[0].Neurons.Count];
    int ipart = 0;
    foreach (string part in parts)
    {
        for (int j = 0; j < part.Length; j++)
            inputs[ipart, j] = part[j] == '1' ? 1.0 : 0.0;
        ipart++;
    }
    double[] result = new double[parts.Length];
    for (int i = 0; i < parts.Length; i++) { double[] inVec = new double[net.Layers[0].Neurons.Count]; for (int j = 0; j < inVec.Length; j++) inVec[j] = inputs[i, j]; double[] outVec = net.Predict(inVec); int best = 0; for (int k = 1; k < outVec.Length; k++) if (outVec[k] > outVec[best]) best = k; result[i] = best; }

    int iSumme = 0;
    for (int i = 0; i < parts.Length; i++)
        iSumme += (int)Math.Pow((double)result[i], (double)(4 * i));

    Console.WriteLine(iSumme);
}

public class Neuron
{
    public double[] _weights;
    public double _bias = 0.0;
    public double LastActivation { get; private set; }

    public Neuron(int inputNumber)
    {
        var rand = new Random();
        _weights = new double[inputNumber];
        _bias = rand.NextDouble() * 2 - 1;
        for (int i = 0; i < inputNumber; i++)
            _weights[i] = rand.NextDouble() * 2 - 1;
    }

    public double Activate(double[] inputs)
    {
        if (inputs.Length != _weights.Length)
            throw new ArgumentException("Not the right Amount of inputs");
        var sum = 0.0;
        for (int i = 0; i < _weights.Length; i++)
            sum += inputs[i] * _weights[i];
        sum += _bias;
        LastActivation = Sigmoid(sum);
        return LastActivation;
    }

    private double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }

    public double SigmoidDerivative()
    {
        return LastActivation * (1 - LastActivation);
    }

    public void UpdateWeights(double[] inputs, double error, double learningrate)
    {
        double delta = error * SigmoidDerivative();
        for (int i = 0; i < _weights.Length; i++)
            _weights[i] += learningrate * delta * inputs[i];
        _bias += learningrate * delta;
    }
}

public class Layer
{
    public List<Neuron> Neurons;

    public Layer(int neuronCount, int inputCount)
    {
        Neurons = new List<Neuron>();
        for (int i = 0; i < neuronCount; i++)
            Neurons.Add(new Neuron(inputCount));
    }

    public double[] Activate(double[] inputs)
    {
        double[] outputs = new double[Neurons.Count];
        for (int i = 0; i < Neurons.Count; i++)
            outputs[i] = Neurons[i].Activate(inputs);
        return outputs;
    }

    public void Train(double[] inputs, double[] errors, double learningRate)
    {
        for (int i = 0; i < Neurons.Count; i++)
            Neurons[i].UpdateWeights(inputs, errors[i], learningRate);
    }
}

public class NeuralNetwork
{
    public List<Layer> Layers;

    public NeuralNetwork(int[] layerStructure)
    {
        Layers = new List<Layer>();
        for (int i = 0; i < layerStructure.Length - 1; i++)
        {
            int inputCount = layerStructure[i];
            int neuronCount = layerStructure[i + 1];
            Layers.Add(new Layer(neuronCount, inputCount));
        }
    }

    public double[] Predict(double[] inputs)
    {
        double[] output = inputs;
        foreach (var layer in Layers)
            output = layer.Activate(output);
        return output;
    }

    public void Train(double[] inputs, double[] expectedOutputs, double learningRate)
    {
        List<double[]> inputArrays = new List<double[]>();
        double[] currentInputs = inputs;
        double[] errorsOutput = new double[expectedOutputs.Length];

        inputArrays.Add(currentInputs);
        foreach (var layer in Layers)
        {
            currentInputs = layer.Activate(currentInputs);
            inputArrays.Add(currentInputs);
        }

        for (int i = 0; i < expectedOutputs.Length; i++)
            errorsOutput[i] = expectedOutputs[i] - inputArrays[inputArrays.Count - 1][i];

        double[] errorsA = errorsOutput;

        for (int layerIndex = Layers.Count - 1; layerIndex >= 0; layerIndex--)
        {
            Layer layer = Layers[layerIndex];
            layer.Train(inputArrays[layerIndex], errorsA, learningRate);

            if (layerIndex > 0)
            {
                double[] nextErrors = new double[Layers[layerIndex - 1].Neurons.Count];
                for (int i = 0; i < nextErrors.Length; i++)
                {
                    double errorSum = 0.0;
                    for (int j = 0; j < layer.Neurons.Count; j++)
                        errorSum += layer.Neurons[j]._weights[i] * errorsA[j];
                    nextErrors[i] = errorSum;
                }
                errorsA = nextErrors;
            }
        }
    }
}
