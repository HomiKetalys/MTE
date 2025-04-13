//
// Created by kyuliea on 2024/11/1.
//

#include "uart.h"

void mte_usart_init()
{
    usart_config();
}
#define UART_BUFFER_SIZE 512
uint8_t uart_buffer[UART_BUFFER_SIZE];
uint32_t begin=0;
uint32_t end=0;

void usart_config(void)
{
    /* enable GPIO clock */
    rcu_periph_clock_enable(EVAL_COM_GPIO_CLK);

    /* enable USART clock */
    rcu_periph_clock_enable(EVAL_COM_CLK);

    /* connect port to USART0 TX */
    gpio_af_set(EVAL_COM_GPIO_PORT, EVAL_COM_AF, EVAL_COM_TX_PIN);

    /* connect port to USART0 RX */
    gpio_af_set(EVAL_COM_GPIO_PORT, EVAL_COM_AF, EVAL_COM_RX_PIN);

    /* configure USART TX as alternate function push-pull */
    gpio_mode_set(EVAL_COM_GPIO_PORT, GPIO_MODE_AF, GPIO_PUPD_PULLUP, EVAL_COM_TX_PIN);
    gpio_output_options_set(EVAL_COM_GPIO_PORT, GPIO_OTYPE_PP, GPIO_OSPEED_60MHZ, EVAL_COM_TX_PIN);

    /* configure USART RX as alternate function push-pull */
    gpio_mode_set(EVAL_COM_GPIO_PORT, GPIO_MODE_AF, GPIO_PUPD_PULLUP, EVAL_COM_RX_PIN);
    gpio_output_options_set(EVAL_COM_GPIO_PORT, GPIO_OTYPE_PP, GPIO_OSPEED_60MHZ, EVAL_COM_RX_PIN);

    /* USART configure */
    usart_deinit(EVAL_COM);
    usart_word_length_set(EVAL_COM, USART_WL_8BIT);
    usart_stop_bit_set(EVAL_COM, USART_STB_1BIT);
    usart_parity_config(EVAL_COM, USART_PM_NONE);
    usart_baudrate_set(EVAL_COM, 115200U);
    usart_receive_config(EVAL_COM, USART_RECEIVE_ENABLE);
    usart_transmit_config(EVAL_COM, USART_TRANSMIT_ENABLE);
    usart_interrupt_enable(EVAL_COM,USART_INT_RBNE);
    usart_enable(EVAL_COM);
    nvic_irq_enable(USART0_IRQn,0,0);
}

void USART0_IRQHandler()
{
    if(usart_interrupt_flag_get(EVAL_COM,USART_INT_RBNE)!=RESET)
    {
        usart_interrupt_flag_clear(EVAL_COM,USART_INT_RBNE);
        uart_buffer[end++]=usart_data_receive(EVAL_COM);
        end=end%UART_BUFFER_SIZE;
        begin=begin!=end?begin:begin+1;
        begin=begin%UART_BUFFER_SIZE;
    }
}

void mte_uart_read(uint8_t* dst,uint32_t byte_size)
{
    uint32_t rev_size=0;
    while(rev_size<byte_size)
    {
        __disable_irq();
        if(begin!=end)
        {
            dst[rev_size++]=uart_buffer[begin++];
            begin=begin%UART_BUFFER_SIZE;
        }
        __enable_irq();
    }
}

uint32_t mte_uart_read_all(uint8_t* dst)
{
    uint32_t rev_size=0;
    while(1)
    {
        __disable_irq();
        if(begin!=end)
        {
            dst[rev_size++]=uart_buffer[begin++];
            begin=begin%UART_BUFFER_SIZE;
        }
        else
        {
            __enable_irq();
            break;
        }
        __enable_irq();
    }
    return rev_size;
}

void mte_uart_write(uint8_t* src,uint32_t byte_size)
{
    uint32_t trans_size;
    for (trans_size = 0; trans_size < byte_size; trans_size++)
    {
        usart_data_transmit(EVAL_COM, (src[trans_size]));
        while(RESET == usart_flag_get(EVAL_COM, USART_FLAG_TBE));
    }
}

#if defined(__CC_ARM)
/* retarget the C library printf function to the USART */
int fputc(int ch, FILE *f)
{
    usart_data_transmit(EVAL_COM, (uint8_t)ch);

    while(RESET == usart_flag_get(EVAL_COM, USART_FLAG_TBE));

    return ch;
}
#elif defined(__GNUC__)
int _write (int fd, char *pBuffer, int size)
{
    int i;
    for (i = 0; i < size; i++)
    {
        usart_data_transmit(EVAL_COM, (uint8_t)(*(pBuffer+i)));
        while(RESET == usart_flag_get(EVAL_COM, USART_FLAG_TBE));
    }
    return i;
}
#else
#warning "Unsupported compiler, printf will be disabled"
#endif


