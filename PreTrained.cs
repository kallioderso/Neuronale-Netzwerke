public class PreTrained
{
    //Variables
    NeuralNetwork _net;
    double _learningRate = { get; private set; };

    //Constructors
    public PreTrained(int[] NeuronsPerLayer) { _net = new NeuralNetwork(NeuronsPerLayer); }

    //Public Methods for DualToDecimal Training
    public double LearningRate() => _learningRate;
    public void SetLearningRate(double learningRate) { _learningRate =  learningRate}
    public void TrainDualToDecimal() => TDTD;
    public int DualToDecimal(string input) => DTD(input);

    //Private Methods for DualToDecimal Training
    private void TDTD()
    {
        while (!CheckDualToDecimalProgress(_net))
        {
            _net.Train(new double[] { 0, 0, 0, 0 }, new double[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, learningRate);
            _net.Train(new double[] { 0, 0, 0, 1 }, new double[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, learningRate);
            _net.Train(new double[] { 0, 0, 1, 0 }, new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, learningRate);
            _net.Train(new double[] { 0, 0, 1, 1 }, new double[] { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, learningRate);
            _net.Train(new double[] { 0, 1, 0, 0 }, new double[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, learningRate);
            _net.Train(new double[] { 0, 1, 0, 1 }, new double[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, learningRate);
            _net.Train(new double[] { 0, 1, 1, 0 }, new double[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, learningRate);
            _net.Train(new double[] { 0, 1, 1, 1 }, new double[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 }, learningRate);
            _net.Train(new double[] { 1, 0, 0, 0 }, new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 }, learningRate);
            _net.Train(new double[] { 1, 0, 0, 1 }, new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 }, learningRate);
            _net.Train(new double[] { 1, 0, 1, 0 }, new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 }, learningRate);
            _net.Train(new double[] { 1, 0, 1, 1 }, new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 }, learningRate);
            _net.Train(new double[] { 1, 1, 0, 0 }, new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 }, learningRate);
            _net.Train(new double[] { 1, 1, 0, 1 }, new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 }, learningRate);
            _net.Train(new double[] { 1, 1, 1, 0 }, new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 }, learningRate);
            _net.Train(new double[] { 1, 1, 1, 1 }, new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, learningRate);
        }
    }

    private int GetArraysHighestPrediction(double[] ergebnise) => Array.IndexOf(ergebnise, ergebnise.Max());
    private bool CheckDualToDecimalProgress(NeuralNetwork net) =>  ((GetArraysHighestPrediction(net.Predict(new double[] { 0, 0, 0, 0 })) == 0) && (GetArraysHighestPrediction(net.Predict(new double[] { 0, 0, 0, 1 })) == 1) && (GetArraysHighestPrediction(net.Predict(new double[] { 0, 0, 1, 0 })) == 2) && (GetArraysHighestPrediction(net.Predict(new double[] { 0, 0, 1, 1 })) == 3) && (GetArraysHighestPrediction(net.Predict(new double[] { 0, 1, 0, 0 })) == 4) && (GetArraysHighestPrediction(net.Predict(new double[] { 0, 1, 0, 1 })) == 5) && (GetArraysHighestPrediction(net.Predict(new double[] { 0, 1, 1, 0 })) == 6) && (GetArraysHighestPrediction(net.Predict(new double[] { 0, 1, 1, 1 })) == 7) && (GetArraysHighestPrediction(net.Predict(new double[] { 1, 0, 0, 0 })) == 8) && (GetArraysHighestPrediction(net.Predict(new double[] { 1, 0, 0, 1 })) == 9) && (GetArraysHighestPrediction(net.Predict(new double[] { 1, 0, 1, 0 })) == 10) && (GetArraysHighestPrediction(net.Predict(new double[] { 1, 0, 1, 1 })) == 11) && (GetArraysHighestPrediction(net.Predict(new double[] { 1, 1, 0, 0 })) == 12) && (GetArraysHighestPrediction(net.Predict(new double[] { 1, 1, 0, 1 })) == 13) && (GetArraysHighestPrediction(net.Predict(new double[] { 1, 1, 1, 0 })) == 14) && (GetArraysHighestPrediction(net.Predict(new double[] { 1, 1, 1, 1 })) == 15));
    private int DTD(string input)
    {
        string[] parts = input.Split(' ');
        int iSumme = 0;
        for(int partIndex = 0; partIndex < parts.Length; partIndex++)
        {
            string part = parts[partIndex];
            double[] inputs = new double[4];
            for(int i = 0; i < Math.Min(part.Length, 4); i++)
                inputs[i] = part[i] == '1' ? 1.0 : 0.0; 
            
            double[] prediction = net.Predict(inputs);
            int digit = iNumber(prediction);
            
            int position = parts.Length - 1 - partIndex; // Stellenwert von rechts
            iSumme += digit * (int)Math.Pow(16, position); // Basis 16 für 4-Bit Gruppen
        }
        return iSumme;
    }
}