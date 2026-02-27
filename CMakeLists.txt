cmake_minimum_required(VERSION 3.20)
project(hms_shared VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Dependencies (guarded for FetchContent consumption) ──────────────────────

if(NOT TARGET spdlog::spdlog)
    find_package(spdlog CONFIG REQUIRED)
endif()

if(NOT TARGET yaml-cpp::yaml-cpp)
    find_package(yaml-cpp CONFIG REQUIRED)
endif()

if(NOT TARGET nlohmann_json::nlohmann_json)
    find_package(nlohmann_json CONFIG REQUIRED)
endif()

if(NOT TARGET PkgConfig::pqxx)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(pqxx REQUIRED IMPORTED_TARGET libpqxx)
endif()

if(NOT TARGET eclipse-paho-mqtt-c::paho-mqtt3as)
    find_package(eclipse-paho-mqtt-c REQUIRED)
endif()

if(NOT TARGET PahoMqttCpp::paho-mqttpp3)
    find_package(PahoMqttCpp REQUIRED)
endif()

# ── Static library ───────────────────────────────────────────────────────────

add_library(hms_shared STATIC
    config/src/config_manager.cpp
    common/src/time_utils.cpp
    db/src/db_pool.cpp
    db/src/api_queries.cpp
    db/src/event_logger.cpp
    mqtt/src/mqtt_client.cpp
)

target_include_directories(hms_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/config/include
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/db/include
    ${CMAKE_CURRENT_SOURCE_DIR}/mqtt/include
)

target_link_libraries(hms_shared PUBLIC
    spdlog::spdlog
    yaml-cpp::yaml-cpp
    nlohmann_json::nlohmann_json
    PkgConfig::pqxx
    eclipse-paho-mqtt-c::paho-mqtt3as
    PahoMqttCpp::paho-mqttpp3
)

# ── Tests (ON when standalone, OFF when consumed) ────────────────────────────

option(HMS_SHARED_BUILD_TESTS "Build hms_shared unit tests"
       ${PROJECT_IS_TOP_LEVEL})

if(HMS_SHARED_BUILD_TESTS)
    enable_testing()

    if(NOT TARGET Catch2::Catch2WithMain)
        find_package(Catch2 3 CONFIG REQUIRED)
    endif()

    include(CTest)
    include(Catch)

    add_executable(shared_tests
        config/tests/config_manager_test.cpp
        common/tests/time_utils_test.cpp
        db/tests/api_queries_test.cpp
    )

    target_link_libraries(shared_tests PRIVATE
        hms_shared
        Catch2::Catch2WithMain
    )

    catch_discover_tests(shared_tests)
endif()
