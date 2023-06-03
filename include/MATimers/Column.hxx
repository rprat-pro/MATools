#pragma once
#include <iostream>

// variables 
#ifdef __MPI
const size_t cWidth = 20; // Column width
const size_t nColumns=6; // Number of columns
const std::string cName[nColumns]={"number Of calls","min (s)", "mean (s)", "max (s)" ,"time ratio (%)", "imb (%)"}; // [1-Imax/Imean]% 
#else
const size_t cWidth =20; //< Column width
const size_t nColumns=3; //< Number of columns
const std::string cName[nColumns]={"number Of calls", "duration (s)" ,"time ratio (%)"}; //< Column names
#endif /* __MPI */
