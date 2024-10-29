//
// Created by kyuliea on 2024/10/29.
//

#ifndef MTE_TOOLS_H
#define MTE_TOOLS_H

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))


#define REPEAT_ALIGN_ID_1(GEN) GEN(0)
#define REPEAT_ALIGN_ID_2(GEN) \
    REPEAT_ALIGN_ID_1(GEN);    \
    GEN(1)
#define REPEAT_ALIGN_ID_3(GEN) \
    REPEAT_ALIGN_ID_2(GEN);    \
    GEN(2)
#define REPEAT_ALIGN_ID_4(GEN) \
    REPEAT_ALIGN_ID_3(GEN);    \
    GEN(3)
#define REPEAT_ALIGN_ID_5(GEN) \
    REPEAT_ALIGN_ID_4(GEN);    \
    GEN(4)
#define REPEAT_ALIGN_ID_6(GEN) \
    REPEAT_ALIGN_ID_5(GEN);    \
    GEN(5)
#define REPEAT_ALIGN_ID_7(GEN) \
    REPEAT_ALIGN_ID_6(GEN);    \
    GEN(6)
#define REPEAT_ALIGN_ID_8(GEN) \
    REPEAT_ALIGN_ID_7(GEN);    \
    GEN(7)

#define REPEAT_ALIGN_ID(N, GEN) REPEAT_ALIGN_ID_##N(GEN)


#define REPEAT_ALIGN_2_ID_C1(n, GEN) GEN(n, 0)
#define REPEAT_ALIGN_2_ID_C2(n, GEN) \
    REPEAT_ALIGN_2_ID_C1(n, GEN);    \
    GEN(n, 1)
#define REPEAT_ALIGN_2_ID_C3(n, GEN) \
    REPEAT_ALIGN_2_ID_C2(n, GEN);    \
    GEN(n, 2)
#define REPEAT_ALIGN_2_ID_C4(n, GEN) \
    REPEAT_ALIGN_2_ID_C3(n, GEN);    \
    GEN(n, 3)
#define REPEAT_ALIGN_2_ID_C5(n, GEN) \
    REPEAT_ALIGN_2_ID_C4(n, GEN);    \
    GEN(n, 4)
#define REPEAT_ALIGN_2_ID_C6(n, GEN) \
    REPEAT_ALIGN_2_ID_C5(n, GEN);    \
    GEN(n, 5)
#define REPEAT_ALIGN_2_ID_C7(n, GEN) \
    REPEAT_ALIGN_2_ID_C6(n, GEN);    \
    GEN(n, 6)
#define REPEAT_ALIGN_2_ID_C8(n, GEN) \
    REPEAT_ALIGN_2_ID_C7(n, GEN);    \
    GEN(n, 7)

#define REPEAT_ALIGN_2_ID_C(n, M, GEN) REPEAT_ALIGN_2_ID_C##M(n, GEN)


#define REPEAT_ALIGN_2_ID_R1(M, GEN) REPEAT_ALIGN_2_ID_C(0, M, GEN)
#define REPEAT_ALIGN_2_ID_R2(M, GEN) \
    REPEAT_ALIGN_2_ID_R1(M, GEN);    \
    REPEAT_ALIGN_2_ID_C(1, M, GEN)
#define REPEAT_ALIGN_2_ID_R3(M, GEN) \
    REPEAT_ALIGN_2_ID_R2(M, GEN);    \
    REPEAT_ALIGN_2_ID_C(2, M, GEN)
#define REPEAT_ALIGN_2_ID_R4(M, GEN) \
    REPEAT_ALIGN_2_ID_R3(M, GEN);    \
    REPEAT_ALIGN_2_ID_C(3, M, GEN)
#define REPEAT_ALIGN_2_ID_R5(M, GEN) \
    REPEAT_ALIGN_2_ID_R4(M, GEN);    \
    REPEAT_ALIGN_2_ID_C(4, M, GEN)
#define REPEAT_ALIGN_2_ID_R6(M, GEN) \
    REPEAT_ALIGN_2_ID_R5(M, GEN);    \
    REPEAT_ALIGN_2_ID_C(5, M, GEN)
#define REPEAT_ALIGN_2_ID_R7(M, GEN) \
    REPEAT_ALIGN_2_ID_R6(M, GEN);    \
    REPEAT_ALIGN_2_ID_C(6, M, GEN)
#define REPEAT_ALIGN_2_ID_R8(M, GEN) \
    REPEAT_ALIGN_2_ID_R7(M, GEN);    \
    REPEAT_ALIGN_2_ID_C(7, M, GEN)

#define REPEAT_ALIGN_2_ID(N, M, GEN) REPEAT_ALIGN_2_ID_R##N(M, GEN)

#define REPEAT_ALIGN_2_ID_R1_A(M, GEN, AO) \
    REPEAT_ALIGN_2_ID_C(0, M, GEN);        \
    AO
#define REPEAT_ALIGN_2_ID_R2_A(M, GEN, AO) \
    REPEAT_ALIGN_2_ID_R1_A(M, GEN,AO);        \
    REPEAT_ALIGN_2_ID_C(1, M, GEN);        \
    AO
#define REPEAT_ALIGN_2_ID_R3_A(M, GEN, AO) \
    REPEAT_ALIGN_2_ID_R2_A(M, GEN,AO);        \
    REPEAT_ALIGN_2_ID_C(2, M, GEN);        \
    AO
#define REPEAT_ALIGN_2_ID_R4_A(M, GEN, AO) \
    REPEAT_ALIGN_2_ID_R3_A(M, GEN,AO);        \
    REPEAT_ALIGN_2_ID_C(3, M, GEN);        \
    AO
#define REPEAT_ALIGN_2_ID_R5_A(M, GEN, AO) \
    REPEAT_ALIGN_2_ID_R4_A(M, GEN,AO);        \
    REPEAT_ALIGN_2_ID_C(4, M, GEN);        \
    AO
#define REPEAT_ALIGN_2_ID_R6_A(M, GEN, AO) \
    REPEAT_ALIGN_2_ID_R5_A(M, GEN,AO);        \
    REPEAT_ALIGN_2_ID_C(5, M, GEN);        \
    AO
#define REPEAT_ALIGN_2_ID_R7_A(M, GEN, AO) \
    REPEAT_ALIGN_2_ID_R6_A(M, GEN,AO);        \
    REPEAT_ALIGN_2_ID_C(6, M, GEN);        \
    AO
#define REPEAT_ALIGN_2_ID_R8_A(M, GEN, AO) \
    REPEAT_ALIGN_2_ID_R7_A(M, GEN,AO);        \
    REPEAT_ALIGN_2_ID_C(7, M, GEN);        \
    AO

#define REPEAT_ALIGN_2_ID_A(N, M, GEN, AO) REPEAT_ALIGN_2_ID_R##N##_A(M, GEN, AO)

#endif//MTE_TOOLS_H
