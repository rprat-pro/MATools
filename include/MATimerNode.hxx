#pragma once

#include <chrono>
#include <vector>
#include <cassert>
#include <EnumTimer.hxx>
#include <Column.hxx>
#include <Output.hxx>
#include <MATimerMPI.hxx>

namespace MATimer
{
	namespace timers
	{
		class MATimerNode
		{
			using duration = std::chrono::duration<double>;

			public:

			MATimerNode();
			MATimerNode(std::string name, MATimerNode* mother);
			MATimerNode* find(std::string name);

			// printer functions
			//
			void print_replicate(size_t begin, size_t end, std::string motif);
			void space();
			void column();
			void end_line();
			void print_banner(size_t shift);
			void print_ending(size_t shift);
			duration* get_ptr_duration();
			void print(size_t shift, double runtime);

			// accessor
			//
			std::string get_name();
			std::size_t get_iteration();
			std::size_t get_level();
			std::vector<MATimerNode*>& get_daughter();
			MATimerNode* get_mother();
			double get_duration();


			private:

			std::string m_name;
			std::size_t m_iteration;
			std::size_t m_level;
			std::vector<MATimerNode*> m_daughter;
			MATimerNode* m_mother;
			duration m_duration;
		};

		template<enumTimer T>
			static MATimerNode*& get_MATimer_node()
			{
				static MATimerNode* __current;
				return __current;
			}
	};
};
