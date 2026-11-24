#include "autoTest.h"
#include "classic.h"
#include "elite-gann.h"
#include "island.h"

#include <iostream>

using namespace std;

int repeatN = 3;

int main() {
  double islandSpend;
  double classicSpend;
  double egannSpend;
  double islandAcc;
  double classicAcc;
  double egannAcc;

  double islandTotal = 0;
  double classicTotal = 0;
  double egannTotal = 0;
  double islandAccTotal = 0;
  double classicAccTotal = 0;
  double egannAccTotal = 0;

  for (int i = 0; i < repeatN; i++) {
    islandTotal += island::startIsland();
    islandAccTotal += test("model-island.json");
  }
  islandSpend = islandTotal / repeatN;
  islandAcc = islandAccTotal / repeatN;

  for (int i = 0; i < repeatN; i++) {
    classicTotal += classic::startClassic();
    classicAccTotal += test("model-classic.json");
  }
  classicSpend = classicTotal / repeatN;
  classicAcc = classicAccTotal / repeatN;

  for (int i = 0; i < repeatN; i++) {
    egannTotal += EGANN::startEGANN();
    egannAccTotal += test("model-egann.json");
  }
  egannSpend = egannTotal / repeatN;
  egannAcc = egannAccTotal / repeatN;

  cout << endl << endl << "Results: " << endl;
  cout << "island: " << islandSpend << "s" << " accuracy: " << islandAcc
       << endl;
  cout << "classic: " << classicSpend << "s" << " accuracy: " << classicAcc
       << endl;
  cout << "egann: " << egannSpend << "s" << " accuracy: " << egannAcc << endl;

  return 0;
}
