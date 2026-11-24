#pragma once

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

namespace classic {
double startClassic();
}; // namespace classic
