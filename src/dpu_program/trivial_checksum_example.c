#include <mram.h>
#include <stdbool.h>
#include <stdint.h>

#define CACHE_SIZE 256
#define BUFFER_SIZE (1 << 16)

__mram_noinit uint8_t buffer[BUFFER_SIZE];
__host uint32_t checksum;

int main() {
  __dma_aligned uint8_t local_cache[CACHE_SIZE];
  checksum = 0;

  for (unsigned int bytes_read = 0; bytes_read < BUFFER_SIZE;) {
    mram_read(&buffer[bytes_read], local_cache, CACHE_SIZE);

    for (unsigned int byte_index = 0; (byte_index < CACHE_SIZE) && (bytes_read < BUFFER_SIZE); byte_index++, bytes_read++) {
      checksum += (uint32_t)local_cache[byte_index];
    }
  }

  return checksum;
}
