//
// Created by kyuliea on 2024/10/30.
//

#include "stdio.h"
#include "string.h"
#include "uart.h"
#include "gd32h7xx_timer.h"
#include "mte_models.h"

uint8_t buffer[128];
uint32_t time0,time1,time;
uint8_t time_str[128];
int8_t mem[MAX_MEM_SIZE];

void timer1_config();

int main(void)
{
    SCB_EnableICache();
    SCB_EnableDCache();
    timer1_config();
    uart_init();
    uint32_t len = 0;
    set_mte_mem_addr(mem);
    int8_t *input_addr;
    float *output_addr;
    while (1) {
//        len += uart_read_all(buffer + len);
//        buffer[len] = '\0';
//        uint8_t *p = strstr(buffer, "\r\n");
//        if (p != NULL) {
//            uint8_t *q=strstr(buffer,"get_time");
//            if(q!=NULL){
//
//            }
//            len = 0;
//        }
        input_addr=get_yolov10_input_addr();
        output_addr=get_yolov10_output_addr();
        int32_t i,j;
        for(i=0;i<256;i++){
            for(j=0;j<256;j++){
                input_addr[i*256*3+j*3+0]=-70;
                input_addr[i*256*3+j*3+1]=10;
                input_addr[i*256*3+j*3+2]=80;
            }
        }
        time0=timer_counter_read(TIMER1);
        yolov10();
        time1=timer_counter_read(TIMER1);
        time=time1-time0;

//        SCB_CleanDCache();
        sprintf(time_str,"infer_time:%lu.%03lums\r\n",time/1000,time%1000);
        uart_write(time_str, strlen(time_str));
    }
    return 0;
}

void timer1_config(void)
{
    /* TIMER0 configuration: generate PWM signals with different duty cycles:
       TIMER0CLK = 300MHz / (299+1) = 1MHz */

    timer_oc_parameter_struct timer_ocintpara;
    timer_parameter_struct timer_initpara;

    rcu_periph_clock_enable(RCU_TIMER1);
    timer_deinit(TIMER1);

    /* TIMER0 configuration */
    timer_initpara.prescaler         = 299;
    //    timer_initpara.alignedmode       = TIMER_COUNTER_EDGE;
    timer_initpara.counterdirection  = TIMER_COUNTER_UP;
    timer_initpara.period            = 0xffffffff;
    timer_initpara.clockdivision     = TIMER_CKDIV_DIV1;
    //    timer_initpara.repetitioncounter = 0;
    timer_init(TIMER1, &timer_initpara);

    timer_enable(TIMER1);
}