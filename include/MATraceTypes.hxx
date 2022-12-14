#pragma once

#include <chrono>
#include <fstream>
#include <vector>
#include <cassert>
#include <string.h>
#include <thread>
#include <map>

#include <MATraceColor.hxx>

namespace MATools
{
	namespace MATrace
	{
		using high_resolution_clock = std::chrono::high_resolution_clock;
		using time_point = std::chrono::time_point<high_resolution_clock>;
		struct MATrace_point
		{
			time_point m_time;
			int m_proc;
			MATrace_point();
			time_point& data();
			const time_point data() const;
			void set_proc();
			void set_proc(int a_rank);
			int get_proc() const;
		};

		constexpr int _sstart = 54;
		class vite_event
		{
			public:
				void add(char a_name[64]);
				int operator[] (char a_name[64]);
				void write_items(std::ofstream& a_out);
				void write_colors(std::ofstream& a_out);

				int m_acc = _sstart;
				std::map<std::string,int> m_data;
		};

		class MATrace_section
		{
			public:
				MATrace_section(char a_name[64], const MATrace_point& a_ref, const MATrace_point& a_start, const MATrace_point& a_end);
				void write(std::ofstream& a_out);
				void write(std::ofstream& a_out, vite_event& a_ve);
				void set_proc(int a_rank);

				int m_proc_id;
				double m_start;
				double m_end;
				char m_name[64];
		};
		
		typedef std::vector<MATrace_section> Trace;
	};
};
