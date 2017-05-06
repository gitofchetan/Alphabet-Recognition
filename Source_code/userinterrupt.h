#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>
#include "/home/chetan/tivaware/inc/hw_ints.h"
#include "/home/chetan/tivaware/inc/hw_memmap.h"
#include "/home/chetan/tivaware/driverlib/debug.h"
#include "/home/chetan/tivaware/driverlib/fpu.h"
#include "/home/chetan/tivaware/driverlib/gpio.h"
#include "/home/chetan/tivaware/driverlib/interrupt.h"
#include "/home/chetan/tivaware/driverlib/pin_map.h"
#include "/home/chetan/tivaware/driverlib/rom.h"
#include "/home/chetan/tivaware/driverlib/sysctl.h"
#include "/home/chetan/tivaware/driverlib/uart.h"

void togglerled();
void toggleled();
void onButtonDown(void);
void SW2intinit();
void UARTconfig();
void uartintinit(void);
void myinit();
void displaymasterinit();

extern int startpredict;
extern char image[1024];
extern char deselogo[600];
