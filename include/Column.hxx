#pragma once
#include <iostream>

// variables 
#ifdef __MPI
const size_t cWidth = 5;
const size_t nColumns=6;
const std::string cName[nColumns]={"number Of Calls","min(s)", "mean(s)", "max(s)" ,"part(%)", "imb(%)"}; // [1-Imax/Imean]% 
#else
const size_t cWidth =20;
const size_t nColumns=3;
const std::string cName[nColumns]={"number Of Calls", "max(s)," ,"part(%)"};
#endif
