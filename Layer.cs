using Neuron;

public class Layer
{
    //Variables
    public List<Neuron> Neurons = {get, private set};

    //Constructor
    public Layer(int neuronCount, int inputCount)
    {
        Neurons = new List<Neuron>();
        for (int i = 0; i < neuronCount; i++)
            Neurons.Add(new Neuron(inputCount));
    }

    //Public Methods
    public double[] Activate(double[] inputs) => Calculate(inputs);
    public void Train(double[] inputs, double[] errors, double learningRate) => Training(inputs, errors, learningRate)
    
    //Private Main Methods
    private double[] Calculate(double[] inputs)
    {
        double[] outputs = new double[Neurons.Count];
        for (int i = 0; i < Neurons.Count; i++)
            outputs[i] = Neurons[i].Activate(inputs);
        return outputs;
    }
    
    private void Training(double[] inputs, double[] errors, double learningRate)
    {
        for (int i = 0; i < Neurons.Count; i++)
            Neurons[i].UpdateWeights(inputs, errors[i], learningRate);
    }
}