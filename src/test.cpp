#include "json.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

using namespace std;
using json = nlohmann::json;

struct Nerve {
  vector<double> weight;
  double offset;
  int id;
};

struct QA {
  vector<double> q;
  double a;
};

inline double relu(double x) { return (x > 0) ? x : 0; }
inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

inline double calNerveOutput(const vector<double> &input, const Nerve &nerve) {
  double sum = nerve.offset;
  for (size_t i = 0; i < input.size(); ++i)
    sum += input[i] * nerve.weight[i];
  return sum;
}

inline double forward(const vector<vector<Nerve>> &net,
                      const vector<double> &input,
                      const vector<int> &layerSizes) {
  vector<double> cur = input;
  for (size_t l = 1; l < net.size(); ++l) {
    vector<double> next;
    for (int j = 0; j < layerSizes[l]; ++j) {
      double raw = calNerveOutput(cur, net[l][j]);
      double act = (l == net.size() - 1) ? sigmoid(raw) : relu(raw);
      next.push_back(act);
    }
    cur = std::move(next);
  }
  return cur[0];
}

inline vector<vector<Nerve>> loadModel(const string &path) {
  ifstream reader(path);
  json data;
  reader >> data;
  reader.close();

  int numLayers = data["layer"]["num"];
  vector<int> layerSizes;
  for (auto &s : data["layer"]["sizes"])
    layerSizes.push_back(s.get<int>());

  vector<vector<Nerve>> net(numLayers);
  for (int l = 0; l < numLayers; ++l) {
    json layerData = data["neurons"][l];
    for (int j = 0; j < layerData.size(); ++j) {
      Nerve n;
      n.id = layerData[j]["id"];
      n.offset = layerData[j]["offset"];
      for (auto &w : layerData[j]["weights"])
        n.weight.push_back(w.get<double>());
      net[l].push_back(n);
    }
  }
  return net;
}

int main(int argc, char *argv[]) {
  string modelPath = "model.json";
  if (argc >= 2)
    modelPath = argv[1];

  vector<vector<Nerve>> net = loadModel(modelPath);
  cout << "Model loaded from " << modelPath << endl;

  vector<int> layerSizes = {};
  for (int i = 0; i < net.size(); i++) {
    layerSizes.push_back(net[i].size());
  }

  vector<QA> testData = {
      {{0.28, 0.10}, 0.0},   {{-0.22, 0.18}, 0.0},  {{-0.31, -0.02}, 0.0},
      {{0.04, -0.29}, 0.0},  {{0.33, -0.09}, 0.0},  {{0.75, 0.00}, 1.0},
      {{-0.41, 0.70}, 1.0},  {{-0.82, -0.35}, 1.0}, {{0.63, -0.55}, 1.0},
      {{-0.21, -0.88}, 1.0},
  };

  double totalLoss = 0.0;
  int correct = 0;
  for (const auto &sample : testData) {
    double pred = forward(net, sample.q, layerSizes);
    double loss = (pred - sample.a) * (pred - sample.a);
    totalLoss += loss;
    bool predictedClass = (pred > 0.5);
    bool actualClass = (sample.a > 0.5);
    if (predictedClass == actualClass)
      ++correct;
  }

  double avgLoss = totalLoss / testData.size();
  double accuracy = 100.0 * correct / testData.size();
  cout << "\nTest Results" << endl;
  cout << "Average MSE Loss: " << avgLoss << endl;
  cout << "Accuracy: " << accuracy << "% (" << correct << "/" << testData.size()
       << ")" << endl;

  return 0;
}
