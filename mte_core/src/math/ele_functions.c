//
// Created by kyuliea on 2025/4/1.
//
#include "mte_core.h"

#define CREATE_ELE_FUNC(NAME, OUTPUT_TYPE, OUTPUT_BYTES)                             \
    void mte_##NAME(                                                                 \
        const int8_t *input, const int32_t ele_nums,                                 \
        const OUTPUT_TYPE *map, OUTPUT_TYPE *map_cache,                              \
        OUTPUT_TYPE *output)                                                         \
    {                                                                                \
        if (map_cache != 0) {                                                        \
            memcpy(map_cache, map, 256 * OUTPUT_BYTES);                              \
            map = map_cache;                                                         \
        }                                                                            \
        const uint8_t *input_p = (const uint8_t *) input;                            \
        const uint8_t *final_input = (const uint8_t *) (input + 4 * (ele_nums / 4)); \
        while (input_p < final_input) {                                              \
            OUTPUT_TYPE m0, m1, m2, m3;                                              \
            m0 = map[input_p[0]];                                                    \
            m1 = map[input_p[1]];                                                    \
            m2 = map[input_p[2]];                                                    \
            m3 = map[input_p[3]];                                                    \
            output[0] = m0;                                                          \
            output[1] = m1;                                                          \
            output[2] = m2;                                                          \
            output[3] = m3;                                                          \
            output += 4;                                                             \
            input_p += 4;                                                            \
        }                                                                            \
        final_input += ele_nums % 4;                                                 \
        while (input_p < final_input) {                                              \
            *output++ = map[*input_p++];                                             \
        }                                                                            \
    }

CREATE_ELE_FUNC(tanh, int8_t, 1)
CREATE_ELE_FUNC(sigmoid, int8_t, 1)
CREATE_ELE_FUNC(quantize, int8_t, 1)
CREATE_ELE_FUNC(dequantize, float, 4)
