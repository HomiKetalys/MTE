cmake_minimum_required(VERSION 3.29)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_COMPILER_FORCED TRUE)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_C_FLAGS "-c -mthumb -gdwarf-2 -MD -O2 -mthumb-interwork -mcpu=cortex-m7 -mthumb -fdata-sections -ffunction-sections -mfloat-abi=hard -mfpu=fpv5-d16")


project(MTE C)
add_library(MTE STATIC
        mte_core/src/mat_mul_kernels.c
        mte_core/inc/mat_mul_kernels.h
        mte_core/inc/mte_core.h
        mte_core/inc/mte_crs.h
        mte_core/inc/mte_tools.h
        mte_core/inc/useful_defines.h)


