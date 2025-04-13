//
// Created by kyuliea on 2024/10/30.
//

#include "stdio.h"
#include "string.h"
#include "uart.h"
#include "timer.h"
#include "mte_models.h"

uint32_t time0,time1,time;
uint8_t str_buffer[128];
int8_t mem[MAX_MEM_SIZE];

void L1_cache_enable()
{
    SCB_EnableICache();
    SCB_EnableDCache();
    SCB->CACR |= (1 << 2);
}

int main(void)
{
    L1_cache_enable();
    mte_timer_init();
    mte_usart_init();
    set_mte_mem_addr(mem);
    int8_t *input_addr;
    float *output_addr;

    while (1) {
        input_addr=get_network_1_input_addr();
        output_addr=get_network_1_output_addr();
        int32_t i,j;
        for(i=0;i<256;i++){
            for(j=0;j<256;j++){
                input_addr[i*256*3+j*3+0]=-70;
                input_addr[i*256*3+j*3+1]=10;
                input_addr[i*256*3+j*3+2]=80;
            }
        }
        time0=mte_get_timer_count();
        network_1();
        time1=mte_get_timer_count();
        time=time1-time0;

        sprintf(str_buffer,"infer_time:%lu.%03lums\r\n",time/1000,time%1000);
        mte_uart_write(str_buffer, strlen(str_buffer));
    }
    return 0;
}

