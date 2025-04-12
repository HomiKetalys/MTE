//
// Created by kyuliea on 2025/4/8.
//

#include "mte_core.h"

void mte_softmax(
    const int8_t *input,const int32_t nums0,const int32_t n,const int32_t nums1,
    const float *map0,float *map0_cache,const float *map1,float *map1_cache,
    int8_t *output,const int32_t output_offset
    ){
    int32_t i,j,k;
    uint8_t *input_p=(uint8_t *)input;
    if(map0_cache!=0){
        memcpy(map0_cache,map0,1024);
        map0=map0_cache;
    }
    if(map1_cache!=0){
        memcpy(map1_cache,map1,1024);
        map1=map1_cache;
    }

    for(i=0;i<nums0;i++){
        for(j=0;j<nums1;j++){
            float sum0=0;
            int32_t idx0=i*n*nums1;
            for(k=0;k<n;k++){
                int32_t idx1=idx0+k*nums1;
                sum0+=map0[input_p[idx1+j]];
            }
            for(k=0;k<n;k++){
                int32_t idx1=idx0+k*nums1;
                float out0=map1[input_p[idx1+j]]/sum0;
                output[idx1+j]=(int8_t)__SSAT8((int32_t)roundf(out0)+output_offset);
            }
        }
    }
}