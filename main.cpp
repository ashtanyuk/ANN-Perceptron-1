
// Многослойный персептрон
// Основано на идеях https://github.com/huangzehao/SimpleNeuralNetwork

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

class TrainingData
{
public:
    TrainingData(const string filename);
    bool isEof(void)
    {
        return m_trainingDataFile.eof();
    }
    void getTopology(vector<unsigned> &topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
    ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned> &topology)
{
    string line;
    string label;

    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;
    if(this->isEof() || label.compare("topology:") != 0)
    {
        abort();
    }

    while(!ss.eof())
    {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }
    return;
}

TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}


unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}


class Neuron;
typedef vector<Neuron> Layer;  // слой нейронов

struct Connection // связи (синапсы)
{
  double weight;  // вес связи
  double delta;
};

class Neuron
{
  private:
    double out;                 // выходное значение
    unsigned index;             // индекс нейрона
    double gradient;
    vector<Connection> weights; // связи нейрона с другими
    static double eta;          // [0.0...1.0] коэффициент обучения
    static double alpha;        // [0.0...n] множитель последнего изменения веса

    static double randomWeight() // случайный вес
    {        
      return rand() / (double)RAND_MAX;
    }
  public:
    // создаем нейрон
    Neuron(unsigned outNums,unsigned index) // количество входов и индекс
    {
      for (unsigned c = 0; c < outNums; ++c){
        weights.push_back(Connection());
        weights.back().weight = randomWeight(); // нейрон получает случайный вес связи
       }
       this->index = index;
    }
    // выходное значение нейрона
    void setOutVal(double outVal)    { out = outVal; }
    double getOutVal()               { return out;   }

    // прямое распространение
    void feedForward(Layer& prev)
    {
       double sum = 0.0;
       // Суммируем выходы с предыдущего слоя
       for (unsigned n = 0; n < prev.size(); ++n)
       {
          sum += prev[n].getOutVal() * prev[n].weights[index].weight;
       }
       // применяем преобразующую функцию
       out = transferFun(sum);
    }

    // преобразующая функция
    static double transferFun(double x)
    {
       return tanh(x);
    }
    double transferFunDeriv(double x) // производная преобразующей функции
    {
       // производная тангенса
       return 1.0 - x * x;
    }
    // изменение входных весов
    void updateInputWeights(Layer &prev)
    {
       for (unsigned n = 0; n < prev.size(); ++n)
       {
          Neuron& neuron = prev[n];
          double  oldDeltaWeight = neuron.weights[index].delta;

          double  newDeltaWeight = eta * neuron.getOutVal() * gradient + alpha * oldDeltaWeight;
          neuron.weights[index].delta = newDeltaWeight;
          neuron.weights[index].weight += newDeltaWeight;
        }
    }
    double sumDOW(const Layer &next) const
    { 
       double sum = 0.0;

       // сумма вклада ошибок
       for (unsigned n = 0; n < next.size() - 1; ++n)
       {
          sum += weights[n].weight * next[n].gradient;
       }
       return sum;
    }
    void calcHiddenGradients(const Layer &next)
    {
       double dow = sumDOW(next);
       gradient = dow * transferFunDeriv(out);
    }
    void calcOutputGradients(double targetVals)
    {
       double delta = targetVals - out;
       gradient = delta * transferFunDeriv(out);
    }
};

double Neuron::eta = 0.75; // overall net learning rate
double Neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight, [0.0..n]

class NNet
{
  private:
    vector<Layer> layers; // слои
    double error;
    double recentAverageError;
    static double recentAverageSmoothingFactor;
  public:

  double getRecentAverageError(void) const { return recentAverageError; }
  NNet(const vector<unsigned>& topology)
  {
     // количество слоев
     unsigned numLayers = topology.size();
     for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
     {
        layers.push_back(Layer());

        // numOutputs для слоя i равен numInputs слоя i+1
        // numOutputs последнего слоя = 0

        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 :topology[layerNum + 1];

        // Создаем новый слой
        for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
        {
            layers.back().push_back(Neuron(numOutputs, neuronNum));
        }

        // Сдвиговый (BIAS) нейрон
        layers.back().back().setOutVal(1.0);
        
    }
   }
   // обратное распространение
   void feedForward(const vector<double>& inputVals)
   {
      
      assert(inputVals.size() == layers[0].size()-1);
      for (unsigned i = 0; i < inputVals.size(); i++)
      {
         layers[0][i].setOutVal(inputVals[i]);
      }
      for (unsigned j = 1; j < layers.size(); j++)
      {
         Layer& prev = layers[j - 1];
         for (unsigned n = 0; n < layers[j].size(); n++)
         {
            layers[j][n].feedForward(prev);
         }
      }
   }
   void backProp(const vector<double>& targetVals)
   {
      // вычисление ошибки сети (RMS ошибок выходных нейронов)
      Layer& output = layers.back();
      error = 0.0;

      for (unsigned n = 0; n < output.size() - 1; ++n)
      {
         double delta = targetVals[n] - output[n].getOutVal();
         error += delta *delta;
      }
      error /= output.size() - 1; // среднее
      error = sqrt(error); // RMS

      recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error)
                           / (recentAverageSmoothingFactor + 1.0);

      // Вычисление градиентов выходного слоя
      for (unsigned n = 0; n < output.size() - 1; ++n)
      {
         output[n].calcOutputGradients(targetVals[n]);
      }

      // Градиенты скрытых слоев
      for (unsigned layerNum = layers.size() - 2; layerNum > 0; --layerNum)
      {
         Layer& hidden = layers[layerNum];
         Layer& next   = layers[layerNum + 1];

         for (unsigned n = 0; n < hidden.size(); ++n)
         {
             hidden[n].calcHiddenGradients(next);
         }
      }

      // For all layers from outputs to first hidden layer,
      // update connection weights

      for (unsigned layerNum = layers.size() - 1; layerNum > 0; --layerNum)
      {
         Layer &layer = layers[layerNum];
         Layer &prev = layers[layerNum - 1];

         for (unsigned n = 0; n < layer.size() - 1; ++n)
         {
            layer[n].updateInputWeights(prev);
         }
       }

   }
   // получение вектора результатов
   void getResults(vector<double>& resultVals)
   { 
      resultVals.clear();
      for (unsigned n = 0; n < layers.back().size() - 1; ++n)
      {
         resultVals.push_back(layers.back()[n].getOutVal());
      }
   }
   
};

double NNet::recentAverageSmoothingFactor = 100.0;


void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for(unsigned i = 0; i < v.size(); ++i)
    {
        cout << v[i] << " ";
    }
    cout << endl;
}

int printResult(double x,double y,NNet& myNet)
{
    vector<double> inputVals,resultVals;
    inputVals.clear();
    inputVals.push_back(x);
    inputVals.push_back(y);
    myNet.feedForward(inputVals);
    myNet.getResults(resultVals);
    int result=(int)(round(resultVals[0]*100)/100*100);
    int expect=(int)(x*10)*(int)(y*10);
    cout<<(int)(x*10)<<"*"<<(int)(y*10)<<"="<<result<<'('<<resultVals[0]<<')'<<endl;
    //showVectorVals("!Outputs:", resultVals);
    return (int)(result!=expect);
}

int main()
{
    TrainingData trainData("trainingData.txt");
    //e.g., {3, 2, 1 }
    vector<unsigned> topology;
    trainData.getTopology(topology);
    topology.clear();
    topology.push_back(2);
    topology.push_back(4);
    topology.push_back(4);
    topology.push_back(4);
    topology.push_back(1);

    //trainData.getTopology(topology);
    NNet myNet(topology);

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;
    while(!trainData.isEof())
    {
        ++trainingPass;
        //cout << endl << "Pass" << trainingPass;

        // Get new input data and feed it forward:
        if(trainData.getNextInputs(inputVals) != topology[0])
            break;
        //showVectorVals(": Inputs :", inputVals);
        myNet.feedForward(inputVals);

        // Collect the net's actual results:
        myNet.getResults(resultVals);
        //showVectorVals("Outputs:", resultVals);

        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(targetVals);
        //showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        // Report how well the training is working, average over recnet
        //cout << "Net recent average error: "
        cout << myNet.getRecentAverageError() << endl;
    }

    //cout << endl << "Training done" << endl;

   int errors=0;
   for(int i=1;i<10;i++)
   {
      for(int j=1;j<10;j++)
      {
        errors+=printResult(i/10.0,j/10.0,myNet);
      }
   }
   cout<<"Errors: "<<errors<<"/100"<<endl;
   return 0;
}

