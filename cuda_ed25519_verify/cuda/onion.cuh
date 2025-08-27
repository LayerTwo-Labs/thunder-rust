#ifndef ONION
#define ONION

#include <stddef.h>

__global__ void onion_address(unsigned char *public_key, uint8_t *checksum);

#endif