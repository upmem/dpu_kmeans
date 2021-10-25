#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include "trees.h"

#define TEXT_BUFFER_SIZE 1000
#define BUFFER_SIZE (1 << 16)

char* call_home(char * filename)
{
    FILE * fPtr;

    char *out;

    char buffer[TEXT_BUFFER_SIZE];
    int totalRead = 0;

    fPtr = fopen(filename, "r");

    if (fPtr == NULL)
    {
        printf("Unable to open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    out = fgets(buffer, TEXT_BUFFER_SIZE, fPtr);

    fclose(fPtr);

    return out;
}

int dpu_test(char* filename) {
  struct dpu_set_t set, dpu;

  const char *DPU_BINARY = filename;

  DPU_ASSERT(dpu_alloc(1, NULL, &set));
  DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));
  DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

  DPU_FOREACH(set, dpu) {
    DPU_ASSERT(dpu_log_read(dpu, stdout));
  }

  DPU_ASSERT(dpu_free(set));

  return 0;
}

void populate_mram(struct dpu_set_t set) {
  uint8_t buffer[BUFFER_SIZE];

  for (int byte_index = 0; byte_index < BUFFER_SIZE; byte_index++) {
    buffer[byte_index] = (uint8_t)byte_index;
  }
  DPU_ASSERT(dpu_broadcast_to(set, "buffer", 0, buffer, BUFFER_SIZE, DPU_XFER_DEFAULT));
}

int checksum(char *filename) {
  struct dpu_set_t set, dpu;
  uint32_t checksum;

  const char *DPU_BINARY = filename;


  DPU_ASSERT(dpu_alloc(1, NULL, &set));
  DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));
  populate_mram(set);

  DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));
  DPU_FOREACH(set, dpu) {
    DPU_ASSERT(dpu_copy_from(dpu, "checksum", 0, (uint8_t *)&checksum, sizeof(checksum)));
    // printf("Computed checksum = 0x%08x\n", checksum);
  }
  DPU_ASSERT(dpu_free(set));
  return checksum;
}
