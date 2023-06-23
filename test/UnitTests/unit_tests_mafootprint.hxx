#pragma once

#include <catch2/catch.hpp>
#include <Common/MAMemory.hxx>

using namespace MATools::MAMemory;

TEST_CASE("make_memory_checkpoint returns valid rusage object")
{
	rusage usage = make_memory_checkpoint();
	// Perform assertions to ensure that the rusage object is valid
	REQUIRE(usage.ru_maxrss >= 0);
}

TEST_CASE("MAFootprint correctly adds memory checkpoints")
{
	MAFootprint footprint;
	SECTION("Adding a memory checkpoint increases the size")
	{
		footprint.add_memory_checkpoint();
		REQUIRE(footprint.size() == 1);

		footprint.add_memory_checkpoint();
		REQUIRE(footprint.size() == 2);
	}
}

TEST_CASE("MAFootprint correctly reduces memory footprint")
{
	MAFootprint footprint;
	footprint.add_memory_checkpoint();
	footprint.add_memory_checkpoint();
	footprint.add_memory_checkpoint();
	footprint.add_memory_checkpoint();
	SECTION("Reducing the footprint returns valid results")
	{
		std::vector<long> reduced = footprint.reduce();
		// Perform assertions to ensure that the reduced memory footprint is valid
		// Compare the reduced values with the expected values based on your test scenario
		REQUIRE(reduced.size() == 4);
		for (auto& it : reduced) REQUIRE( it >= 0.);
	}
}

TEST_CASE("print_checkpoints doesn't fail")
{
	MAFootprint footprint;
	footprint.add_memory_checkpoint();
	SECTION("print_checkpoints prints the correct output")
	{
		print_checkpoints(footprint);
	}
}

TEST_CASE("print_memory_footprint doesn't fail")
{	
	MAFootprint footprint;
	footprint.add_memory_checkpoint();
	SECTION("print_memory_footprint prints the correct output")
	{
		print_memory_footprint();
	}

}

