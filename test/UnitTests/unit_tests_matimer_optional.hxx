#include <MATimers/MATimerOptional.hxx>
#include <MAOutput/MAOutput.hxx>


TEST_CASE("MATimerOptional - Full Tree Mode", "[MATimerOptional]") {
    SECTION("Check initial state") {
        bool isFullTreeMode = MATools::MATimer::Optional::is_full_tree_mode();
        REQUIRE_FALSE(isFullTreeMode);
    }

    SECTION("Activate Full Tree Mode") {
        MATools::MATimer::Optional::active_full_tree_mode();
        bool isFullTreeMode = MATools::MATimer::Optional::is_full_tree_mode();
        REQUIRE(isFullTreeMode);
    }
}

TEST_CASE("MATimerOptional - Print Timetable Mode", "[MATimerOptional]") {
    SECTION("Check initial state") {
        bool isPrintTimetable = MATools::MATimer::Optional::is_print_timetable();
        REQUIRE(isPrintTimetable);
    }

    SECTION("Disable Print Timetable Mode") {
        MATools::MATimer::Optional::disable_print_timetable();
        bool isPrintTimetable = MATools::MATimer::Optional::is_print_timetable();
        REQUIRE_FALSE(isPrintTimetable);
    }

    SECTION("Enable Print Timetable Mode") {
        MATools::MATimer::Optional::enable_print_timetable();
        bool isPrintTimetable = MATools::MATimer::Optional::is_print_timetable();
        REQUIRE(isPrintTimetable);
    }
}

TEST_CASE("MATimerOptional - Write File Mode", "[MATimerOptional]") {
    SECTION("Check initial state") {
        bool isWriteFile = MATools::MATimer::Optional::is_write_file();
        REQUIRE(isWriteFile);
    }

    SECTION("Disable Write File Mode") {
        MATools::MATimer::Optional::disable_write_file();
        bool isWriteFile = MATools::MATimer::Optional::is_write_file();
        REQUIRE_FALSE(isWriteFile);
    }

    SECTION("Enable Write File Mode") {
        MATools::MATimer::Optional::enable_write_file();
        bool isWriteFile = MATools::MATimer::Optional::is_write_file();
        REQUIRE(isWriteFile);
    }
}
