// mining.h
#ifndef MINING_H
#define MINING_H

#include <cuda_runtime.h>

// Declare the CUDA kernel and the host function to mine a block
extern "C" void mine_block(const char* block_data, int block_data_len, int difficulty, unsigned char* valid_hash, int* nonce_found, int max_nonce);

#endif // MINING_H
