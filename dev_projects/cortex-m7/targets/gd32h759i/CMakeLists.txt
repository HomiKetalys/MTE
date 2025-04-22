cmake_minimum_required(VERSION 3.29)

set(MODEL_PATH "../../../../temp/c_codes")
set(MTE_CORE_PATH "../../../../mte_core")

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

add_definitions(-DGD32H7XX)
include_directories(
        ${MTE_CORE_PATH}/inc
        ${MODEL_PATH}
        ../../../CMSIS_5/CMSIS/Core/Include
        lib/inc
        lib
)
file(GLOB_RECURSE DEV_SOURCES "lib/*.c" "lib/*.h")
message("${DEV_SOURCES}")

file(GLOB_RECURSE MTE_SOURCES "${MTE_CORE_PATH}/src/*.c" "${MTE_CORE_PATH}/inc/*.h")
message("${MTE_SOURCES}")

file(GLOB_RECURSE MODEL_SOURCES "${MODEL_PATH}/*.c" "${MODEL_PATH}/*.h")
message("${MODEL_SOURCES}")

foreach(file ${MTE_SOURCES})
    set_source_files_properties(${file} PROPERTIES COMPILE_FLAGS "-O3")
endforeach()

foreach(file ${MODEL_SOURCES})
    set_source_files_properties(${file} PROPERTIES COMPILE_FLAGS "-O3")
endforeach()

set(CMAKE_C_FLAGS "-gdwarf-2 -mcpu=cortex-m7 -mthumb -mthumb-interwork -fdata-sections -ffunction-sections -mfloat-abi=hard -mfpu=fpv5-d16")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--gc-sections -Wl,-Map=memory.map -T${CMAKE_SOURCE_DIR}/flash.ld")

project(cortex-m7 C ASM)
add_executable(cortex-m7
        ../../../main.c
        startup.S
        ${DEV_SOURCES}
        ${MTE_SOURCES}
        ${MODEL_SOURCES}
)
set_target_properties(cortex-m7 PROPERTIES SUFFIX ".elf")

find_program(C_OBJCOPY NAMES objcopy)
find_program(C_OBJDUMP NAMES objdump)

add_custom_command(TARGET cortex-m7 POST_BUILD
        COMMAND ${C_OBJCOPY} -O ihex cortex-m7.elf cortex-m7.hex
        COMMENT "Generating HEX file from ELF file"
)
add_custom_command(TARGET cortex-m7 POST_BUILD
        COMMAND ${C_OBJDUMP} -h cortex-m7.elf
        COMMENT "Displaying section sizes"
)