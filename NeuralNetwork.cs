using Neuron;
using Layer;

public class NeuralNetwork
{
    //Variables
    public List<Layer> Layers = {get, private set};

    //Constructor
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

    //Public Methods
    public double[] Predict(double[] inputs) => CalculateEverything(inputs);
    public void Train(double[] inputs, double[] expectedOutputs, double learningRate) => Training(inputs, expectedOutputs, learningRate);
    
    //Private Main Methods
    private double[] CalculateEverything(double[] inputs)
    {
        double[] output = inputs;
        foreach (var layer in Layers)
            output = layer.Predict(output);
        return output;
    }

    private void Training(double[] inputs, double[] expectedOutputs, double learningRate)
    {
        List<double[]> inputArrays = new List<double[]>();
        double[] currentInputs = inputs;
        double[] errorsOutput = new double[expectedOutputs.Length];

        inputArrays.Add(currentInputs);
        foreach (var layer in Layers)
        {
            currentInputs = layer.Predict(currentInputs);
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