#include <string>

#include <torch/extension.h>

#include "pybind/extern.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("ApproxMatch", &ApproxMatch);
  m.def("MatchCost", &MatchCost);
  m.def("MatchCostGrad", &MatchCostGrad);
  m.def("NNDistance", &NNDistance);
  m.def("NNDistanceGrad", &NNDistanceGrad);
}
