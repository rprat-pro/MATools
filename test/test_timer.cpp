#include <MATools.hxx>
#include <tclap/CmdLine.h>
#include "common.hxx"

int main(int argc, char * argv[]) 
{
	MATools::initialize(&argc,&argv);
	int robust = -1; 
	int tcase = -1;

	try {
		TCLAP::CmdLine cmd("timers tests", ' ', "1.0");
		TCLAP::ValueArg<int> _tcase("c", "case", "test case", false, -2, "int");
		TCLAP::ValueArg<int> _robust("r", "robustness", "test of robustess", false, -2, "int");


		cmd.add(_tcase);
		cmd.add(_robust);

		cmd.parse(argc, argv);

		tcase 	= _tcase.getValue();
		robust 	= _robust.getValue();
	} 
	catch (TCLAP::ArgException& e) 
	{
		std::cerr << "error: " << e.error() << " for argument " << e.argId() << std::endl;
	}

	assert(	
		robust >= 0 
		&& robust < 2 
		&& "bad choice for robust parameter"
	);

	assert(
		tcase >= 0 
		&& tcase < 6 
		&& "bad choice for tcase parameter"
	);


	if(robust)
	{
		constexpr int size = 100;
		launch<size>(tcase);
	}
	else
	{
		constexpr int size = 1;
		launch<size>(tcase);
	}

	MATools::finalize();

	return 0;
}
