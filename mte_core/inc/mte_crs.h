//
// Created by kyuliea on 2024/10/29.
//

#ifndef MTE_MTE_CRS_H
#define MTE_MTE_CRS_H

#include <stdint.h>
#include <string.h>

#if defined(__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050)
#ifndef __ASM
#define __ASM __asm
#endif
#ifndef __INLINE
#define __INLINE __inline
#endif
#ifndef __STATIC_INLINE
#define __STATIC_INLINE static __inline
#endif
#ifndef __STATIC_FORCEINLINE
#define __STATIC_FORCEINLINE __attribute__((always_inline)) static __inline
#endif
#ifndef __RESTRICT
#define __RESTRICT __restrict
#endif

#elif defined(__GNUC__)
#ifndef __ASM
#define __ASM __asm
#endif
#ifndef __INLINE
#define __INLINE inline
#endif
#ifndef __STATIC_INLINE
#define __STATIC_INLINE static inline
#endif
#ifndef __STATIC_FORCEINLINE
#define __STATIC_FORCEINLINE __attribute__((always_inline)) static inline
#endif
#ifndef __RESTRICT
#define __RESTRICT __restrict
#endif
#else
#error "Unsupported compiler. Add support as needed"
#endif

#if defined(__ARM_ARCH_7EM__)
__STATIC_FORCEINLINE int32_t __SXTB16_ROR8(int32_t op1)
{
    int32_t result;
    __ASM volatile("sxtb16 %0, %1, ROR #8" : "=r"(result) : "r"(op1));
    return result;
}

__STATIC_FORCEINLINE int32_t __SXTB16(int32_t op1)
{
    int32_t result;
    __ASM volatile("sxtb16 %0, %1" : "=r"(result) : "r"(op1));
    return result;
}

__STATIC_FORCEINLINE int32_t __SMMLAR(int32_t op1, int32_t op2, int32_t op3)
{
    int32_t result;
    __ASM volatile("smmlar %0, %1, %2, %3" : "=r"(result) : "r"(op1), "r"(op2), "r"(op3));
    return result;
}

__STATIC_FORCEINLINE int32_t __SSAT8(int32_t op1)
{
    int32_t result;
    __ASM volatile("ssat %0, #8, %1" : "=r"(result) : "r"(op1));
    return result;
}

__STATIC_FORCEINLINE int32_t __SMLAD(int32_t op1, int32_t op2, int32_t op3)
{
    int32_t result;
    __ASM volatile("smlad %0, %1, %2, %3" : "=r"(result) : "r"(op1), "r"(op2), "r"(op3));
    return result;
}


__STATIC_FORCEINLINE int32_t __SMLABB(int32_t op1, int32_t op2, int32_t op3)
{
    int32_t result;
    __ASM volatile("smlabb %0, %1, %2, %3" : "=r"(result) : "r"(op1), "r"(op2), "r"(op3));
    return result;
}

__STATIC_FORCEINLINE uint32_t __SADD16(uint32_t op1, uint32_t op2)
{
    uint32_t result;

    __ASM volatile("sadd16 %0, %1, %2" : "=r"(result) : "r"(op1), "r"(op2));
    return (result);
}

#define __PKHBT(ARG1, ARG2, ARG3)                                                              \
    __extension__({                                                                            \
        uint32_t __RES, __ARG1 = (ARG1), __ARG2 = (ARG2);                                      \
        __ASM("pkhbt %0, %1, %2, lsl %3" : "=r"(__RES) : "r"(__ARG1), "r"(__ARG2), "I"(ARG3)); \
        __RES;                                                                                 \
    })

#define __PKHTB(ARG1, ARG2, ARG3)                                                                  \
    __extension__({                                                                                \
        uint32_t __RES, __ARG1 = (ARG1), __ARG2 = (ARG2);                                          \
        if (ARG3 == 0)                                                                             \
            __ASM("pkhtb %0, %1, %2" : "=r"(__RES) : "r"(__ARG1), "r"(__ARG2));                    \
        else                                                                                       \
            __ASM("pkhtb %0, %1, %2, asr %3" : "=r"(__RES) : "r"(__ARG1), "r"(__ARG2), "I"(ARG3)); \
        __RES;                                                                                     \
    })

#endif


__STATIC_FORCEINLINE int32_t mte_read_q15x2_ia(const int16_t **in_q15)
{
    int32_t val;
    memcpy(&val, *in_q15, 4);
    *in_q15 += 2;
    return (val);
}

__STATIC_FORCEINLINE int32_t mte_read_s8x4_ia(const int8_t **in_s8)
{
    int32_t val;
    memcpy(&val, *in_s8, 4);
    *in_s8 += 4;
    return (val);
}
__STATIC_FORCEINLINE void i8x4_to_2xi16x2_offset_reordered_ele_ia(int32_t **src, int32_t **dst, int32_t offset_i16x2)
{
    int32_t out_i16x2_1;
    int32_t out_i16x2_2; /* Read i8x4 from src */
    out_i16x2_1 = *(*src);
    *src +=1; /* Expand the sign of each signed 8-bit integer to 16 bits and rearrange them alternately */
    out_i16x2_2 = __SXTB16(out_i16x2_1);
    out_i16x2_1 = __SXTB16_ROR8(out_i16x2_1);
    out_i16x2_2 = __SADD16(out_i16x2_2, offset_i16x2);
    out_i16x2_1 = __SADD16(out_i16x2_1, offset_i16x2);
    *(*dst) = out_i16x2_2;
    *dst += 1;
    *(*dst) = out_i16x2_1;
    *dst += 1;
}


//#define i8x4_to_2xi16x2_offset_reordered_ele(src, dst, offset_i16x2)                                 \
//    {                                                                                                \
//        int32_t out_i16x2_1;                                                                         \
//        int32_t out_i16x2_2;                                                                         \
//        /* Read i8x4 from src */                                                                     \
//        out_i16x2_1 = *((int32_t *) src);                                                            \
//        src = ((int32_t *) src) + 1;                                                                 \
//        /* Expand the sign of each signed 8-bit integer to 16 bits and rearrange them alternately */ \
//        out_i16x2_2 = __SXTB16(out_i16x2_1);                                                         \
//        out_i16x2_1 = __SXTB16_ROR8(out_i16x2_1);                                                    \
//        out_i16x2_2 = __SADD16(out_i16x2_2, offset_i16x2);                                           \
//        out_i16x2_1 = __SADD16(out_i16x2_1, offset_i16x2);                                           \
//        *((int32_t *) dst) = out_i16x2_2;                                                            \
//        dst = ((int32_t *) dst) + 1;                                                                 \
//        *((int32_t *) dst) = out_i16x2_1;                                                            \
//        dst = ((int32_t *) dst) + 1;                                                                 \
//    }




#endif//MTE_MTE_CRS_H
