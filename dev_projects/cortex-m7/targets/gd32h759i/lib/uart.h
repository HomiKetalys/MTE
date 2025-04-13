//
// Created by kyuliea on 2024/11/1.
//

#ifndef UART_H
#define UART_H

#include "gd32h7xx.h"
#include "gd32h759i_eval.h"

void mte_usart_init();
uint32_t mte_uart_read_all(uint8_t* dst);
void mte_uart_read(uint8_t* dst,uint32_t byte_size);
void mte_uart_write(uint8_t* src,uint32_t byte_size);

#endif//UART_H
