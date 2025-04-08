cmake_minimum_required(VERSION 3.29)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_COMPILER_FORCED TRUE)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

if(NOT CMAKE_BUILD_TYPE)
    message(FATAL_ERROR "CMAKE_BUILD_TYPE variable is not defined!")
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    message("Building in Debug mode.")
else()
    message("Building in Release mode or another non-debug build type.")
    if(${CMAKE_BUILD_TYPE} MATCHES "^(MinSizeRel|RelWithDebInfo)$")
        message("Note: It's a release-like configuration.")
    endif()
endif()

include_directories(
        mte_core/inc
)
file(GLOB_RECURSE SOURCES "mte_core/*.c" "mte_core/*.h")
message("${SOURCES}")
set(CMAKE_C_FLAGS "-gdwarf-2 -mcpu=cortex-m7 -mthumb -mthumb-interwork -fdata-sections -ffunction-sections -mfloat-abi=hard -mfpu=fpv5-d16")

project(MTE C)
add_library(MTE STATIC
        ${SOURCES}
)


