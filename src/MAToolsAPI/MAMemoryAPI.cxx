#include <MAToolsAPI/MAMemoryAPI.hxx>


std::vector<std::string>& get_mem_labels()
{
	static std::vector<std::string> ret;
	return ret;
}
