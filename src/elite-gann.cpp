#include "elite-gann.h"
#include "json.hpp"
#include <chrono>
#include <execution>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <thread>
#include <vector>

using namespace std;
using json = nlohmann::json;

namespace EGANN {
struct QA {
  vector<double> q;
  double a;
};

struct Nerve {
  vector<double> weight;
  double value;
  double offset;
  int id;
};

random_device rd;
mt19937 gen(rd());

int netNum = 16;
int layerNum = 4;
vector<int> layerNerveNumLst = {4, 10, 10, 1};
int hiddenGenNum;
vector<vector<vector<Nerve>>> nerveNetLst;
vector<QA> qaLst = {};
int nowQAID;

double crossoverP = 0.50;
int crossoverParentN = 2;
int tournamentSelectionK = 5;
double mutateP = 0.40;
double mutateSigma = 0.50;
double learnRate = 0.10;
double learnNetP = 0.10;
double selectRatio = 0.10;
double testBatchP = 0.30;
int testBatchSize;

bool isRunning;

// 生成同心圆数据
vector<QA> generateConcentricCircles(int samplesPerClass = 100) {
  vector<QA> data;
  random_device rd;
  mt19937 gen(rd());
  normal_distribution<> noise(0, 0.1);

  // 内圆（半径0.3，标签0）
  for (int i = 0; i < samplesPerClass; ++i) {
    double angle = uniform_real_distribution<>(0, 2 * M_PI)(gen);
    double r = 0.3 + noise(gen);
    data.push_back({{r * cos(angle), r * sin(angle)}, 0.0});
  }
  // 外环（半径0.6~0.9，标签1）
  for (int i = 0; i < samplesPerClass; ++i) {
    double angle = uniform_real_distribution<>(0, 2 * M_PI)(gen);
    double r = uniform_real_distribution<>(0.6, 0.9)(gen) + noise(gen);
    data.push_back({{r * cos(angle), r * sin(angle)}, 1.0});
  }
  return data;
}

void shuffleQALst(vector<QA> &lst) {
  for (int i = 0; i < lst.size() - 1; i++) {
    // 当前和后面一个打乱
    double p = (uniform_real_distribution<double>(-1.0, 1.0))(gen);
    if (p >= 0.0) {
      QA tmp = lst[i];
      lst[i] = lst[i + 1];
      lst[i + 1] = tmp;
    }
  }
}

int rouletteWheelSelection(const vector<double> &fx, mt19937 &gen) {
  double total = 0.0;
  for (double v : fx)
    total += v;

  uniform_real_distribution<double> dist(0.0, total);
  double r = dist(gen);

  double partial = 0.0;
  for (int i = 0; i < fx.size(); ++i) {
    partial += fx[i];
    if (partial >= r)
      return i;
  }

  return fx.size() - 1;
}

vector<int> tournamentSelection(const vector<double> &fx, mt19937 &gen) {
  /*
   * 从当前种群中随机抽取 k 个个体；
   * 比较这 k 个体的适应度；
   * 选出其中最优个体；
   * 重复上述过程，直到选满所需父代数量。
   * */

  vector<int> ret = {};

  while (ret.size() < crossoverParentN) {
    int bestID = -1;
    double bestFitness = numeric_limits<double>::min();
    for (int j = 0; j < tournamentSelectionK; j++) {
      int id = uniform_int_distribution<int>(0, fx.size() - 1)(gen);
      double fitness = fx[id];
      if (fitness > bestFitness) {
        bestFitness = fitness;
        bestID = id;
      }
    }
    if (find(ret.begin(), ret.end(), bestID) == ret.end())
      ret.push_back(bestID);
  }

  return ret;
}

double calNerveOutput(const vector<double> input, const Nerve &nerve) {
  double ret = 0.0;
  for (int i = 0; i < input.size(); i++) {
    ret += input[i] * nerve.weight[i];
  }
  ret += nerve.offset;

  return ret;
}

void mutate(vector<vector<Nerve>> &target) {
  int mutateN = (uniform_int_distribution<int>(0, hiddenGenNum * mutateP))(gen);
  for (int k = 0; k < mutateN; k++) {
    int l = (uniform_int_distribution<int>(1, target.size() - 1))(gen);
    int i = (uniform_int_distribution<int>(0, target[l].size() - 1))(gen);
    bool isWeight = (uniform_int_distribution<int>(0, 1))(gen);
    if (isWeight) {
      int g = (uniform_int_distribution<int>(0, target[l][i].weight.size() -
                                                    1))(gen);
      double offset =
          (uniform_real_distribution<double>(-mutateSigma, mutateSigma))(gen);

      target[l][i].weight[g] += offset;
    } else {
      double offset =
          (uniform_real_distribution<double>(-mutateSigma, mutateSigma))(gen);
      target[l][i].offset += offset;
    }
  }
}

void crossover(const vector<vector<Nerve>> &parent1,
               const vector<vector<Nerve>> &parent2,
               vector<vector<Nerve>> &target) {
  target = parent1;
  int crossoverN =
      (uniform_int_distribution<int>(1, hiddenGenNum * crossoverP))(gen);
  for (int k = 0; k < crossoverN; k++) {
    int l = (uniform_int_distribution<int>(1, parent1.size() - 1))(gen);
    int i = (uniform_int_distribution<int>(0, parent2[l].size() - 1))(gen);
    bool isWeight = (uniform_int_distribution<int>(0, 1))(gen);
    if (isWeight) {
      int g = (uniform_int_distribution<int>(0, parent1[l][i].weight.size() -
                                                    1))(gen);

      target[l][i].weight[g] = parent2[l][i].weight[g];
    } else {
      target[l][i].offset = parent2[l][i].offset;
    }
  }
}

void learnElite(int eliteNetID, int learnerNetID) {
  for (int a = 0; a < selectRatio * hiddenGenNum; a++) {
    int l = (uniform_int_distribution<int>(1, nerveNetLst[eliteNetID].size() -
                                                  1))(gen);
    int i = (uniform_int_distribution<int>(
        0, nerveNetLst[eliteNetID][l].size() - 1))(gen);
    bool isWeight = (uniform_int_distribution<int>(0, 1))(gen);
    if (isWeight) {
      int g = (uniform_int_distribution<int>(
          0, nerveNetLst[eliteNetID][l][i].weight.size() - 1))(gen);

      nerveNetLst[learnerNetID][l][i].weight[g] +=
          (nerveNetLst[eliteNetID][l][i].weight[g] -
           nerveNetLst[learnerNetID][l][i].weight[g]) *
          learnRate;
    } else {
      nerveNetLst[learnerNetID][l][i].offset +=
          (nerveNetLst[eliteNetID][l][i].offset -
           nerveNetLst[learnerNetID][l][i].offset) *
          learnRate;
    }
  }
}

double relu(double x) { return (x > 0) ? x : 0; }

double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

void init() {
  isRunning = true;

  nerveNetLst.clear();
  qaLst.clear();

  qaLst = generateConcentricCircles();

  shuffleQALst(qaLst);

  testBatchSize = qaLst.size() * testBatchP;

  nowQAID = 0;

  int nerveID = 0;
  for (int a = 0; a < netNum; a++) {
    nerveNetLst.push_back({});
    hiddenGenNum = 0;
    nerveID = 0;
    for (int i = 0; i < layerNum; i++) {
      if (i > 0 && i < layerNum - 1) {
        hiddenGenNum += layerNerveNumLst[i - 1] * layerNerveNumLst[i];
      }

      nerveNetLst[a].push_back({});

      for (int j = 0; j < layerNerveNumLst[i]; j++) {
        Nerve nn;
        if (i != 0) {
          nn.weight = {};
          for (int k = 0; k < layerNerveNumLst[i - 1]; k++) {
            double nw = (uniform_real_distribution<double>(-1.0, 1.0))(gen);
            nn.weight.push_back(nw);
          }
        }
        nn.offset = 0.0;
        nn.id = nerveID;
        nerveNetLst[a][i].push_back(nn);
        nerveID++;
      }
    }
  }

  for (int a = 0; a < netNum; ++a) {
    cout << "Net " << a << " first neuron id: " << nerveNetLst[a][0][0].id
         << endl;
  }
}

double forward(int netID, vector<double> input, double answer) {
  double loss;
  vector<vector<double>> layerInputs;
  layerInputs.push_back(input);
  for (int i = 1; i < layerNum; i++) {
    layerInputs.push_back({});
    for (int j = 0; j < layerNerveNumLst[i]; j++) {
      double result =
          calNerveOutput(layerInputs[i - 1], nerveNetLst[netID][i][j]);
      if (i == layerNum - 1)
        result = sigmoid(result);
      else
        result = relu(result);
      layerInputs[i].push_back(result);
    }
  }
  double predict = layerInputs[layerNum - 1][0];
  loss = (predict - answer) * (predict - answer);
  // cout << "netID: " << netID << " predict: " << predict << " answer: " <<
  // answer << endl;
  return loss;
}

double calNetFitness(int netID) {
  double ret;

  double lossSum = 0;
  uniform_int_distribution<> dis(0, qaLst.size() - 1);
  for (int i = 0; i < testBatchSize; i++) {
    int a = dis(gen);
    double loss = forward(netID, qaLst[a].q, qaLst[a].a);
    lossSum += loss;
  }

  ret = 1 - lossSum / testBatchSize;
  return ret;
}

void saveModel(int netID, const string &path) {
  json data;

  data["layer"]["num"] = nerveNetLst[netID].size();
  data["layer"]["sizes"] = json::array();
  for (const auto &layer : nerveNetLst[netID]) {
    data["layer"]["sizes"].push_back(layer.size());
  }

  data["neurons"] = json::array();
  for (const auto &layer : nerveNetLst[netID]) {
    json layerArr = json::array();
    for (const auto &neuron : layer) {
      layerArr.push_back({{"id", neuron.id},
                          {"offset", neuron.offset},
                          {"weights", neuron.weight}});
    }
    data["neurons"].push_back(layerArr);
  }

  ofstream writer(path);
  writer << data.dump(2);
  writer.close();
}

void epoch() {
  if (!isRunning)
    return;

  vector<double> q = qaLst[nowQAID].q;
  double ans = qaLst[nowQAID].a;

  int worstNetID = -1;
  int bestNetID = -1;
  double bestNetFitness = numeric_limits<double>::min();
  double worstNetFitness = numeric_limits<double>::max();
  vector<double> fx = {};
  vector<int> netIDRank(netNum);
  for (int a = 0; a < netNum; a++) {
    double fitness = calNetFitness(a);
    fx.push_back(fitness);
    if (fitness >= 0.99) {
      saveModel(a, "model-egann.json");
      cout << "INFO: Training ended" << endl;
      isRunning = false;
    }
    if (fitness < worstNetFitness) {
      worstNetID = a;
      worstNetFitness = fitness;
    }
    if (fitness > bestNetFitness) {
      bestNetID = a;
      bestNetFitness = fitness;
    }
  }
  iota(netIDRank.begin(), netIDRank.end(), 0);
  sort(netIDRank.begin(), netIDRank.end(),
       [&](int a, int b) { return fx[a] > fx[b]; });

  cout << "the best fitness: " << bestNetFitness << endl;

  // 学习精英
  for (int i = netIDRank.size() - 1; i >= netNum * learnNetP; i--) {
    learnElite(bestNetID, netIDRank[i]);
  }

  // 交叉
  // 锦标赛选择
  vector<int> clst = tournamentSelection(fx, gen);
  vector<vector<Nerve>> newNet;
  for (int i = 0; i < clst.size() - 1; i++) {
    crossover(nerveNetLst[clst[i]], nerveNetLst[clst[i + 1]],
              nerveNetLst[worstNetID]);
  }

  nowQAID++;
  if (nowQAID >= qaLst.size()) {
    nowQAID = 0;
  }

  // 变异
  int mID = uniform_int_distribution<int>(0, layerNum - 1)(gen);
  mutate(nerveNetLst[mID]);
}

double startEGANN() {
  init();

  auto start = chrono::high_resolution_clock::now();
  while (isRunning)
    epoch();
  auto end = chrono::high_resolution_clock::now();
  chrono::duration<double> elapsed = end - start;
  cout << "Training time: " << elapsed.count() << " seconds" << endl;

  return elapsed.count();
}

}; // namespace EGANN
