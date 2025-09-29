public class Neuron
{
    //Variables
    public double[] _weights;
    public double _bias = 0.0;
    public double LastActivation { get; private set; }

    //Constructor
    public Neuron(int inputNumber)
    {
        var rand = new Random();
        _weights = new double[inputNumber];
        _bias = rand.NextDouble() * 2 - 1;
        for (int i = 0; i < inputNumber; i++)
            _weights[i] = rand.NextDouble() * 2 - 1;
    }

    //Public Methods
    public double Activate(double[] inputs) => Calculate(inputs);
    public void UpdateWeights(double[] inputs, double error, double learningrate) => ChangeWeights(inputs, error, learningrate);

    //Private Methods
    private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
    private double SigmoidDerivative() => LastActivation * (1 - LastActivation);

    //Private Main Methods
    private double Calculate(double[] inputs)
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

    private void ChangeWeights(double[] inputs, double error, double learningrate)
    {
        double delta = error * SigmoidDerivative();
        for (int i = 0; i < _weights.Length; i++)
            _weights[i] += learningrate * delta * inputs[i];
        _bias += learningrate * delta;
    }
}