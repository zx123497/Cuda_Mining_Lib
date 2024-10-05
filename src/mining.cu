#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define ROTLEFT(a, b) ((a << b) | (a >> (32 - b)))
#define ROTRIGHT(a, b) ((a >> b) | (a << (32 - b)))

#define CH(x, y, z) ((x & y) ^ (~x & z))
#define MAJ(x, y, z) ((x & y) ^ (x & z) ^ (y & z))
#define EP0(x) (ROTRIGHT(x, 2) ^ ROTRIGHT(x, 13) ^ ROTRIGHT(x, 22))
#define EP1(x) (ROTRIGHT(x, 6) ^ ROTRIGHT(x, 11) ^ ROTRIGHT(x, 25))
#define SIG0(x) (ROTRIGHT(x, 7) ^ ROTRIGHT(x, 18) ^ (x >> 3))
#define SIG1(x) (ROTRIGHT(x, 17) ^ ROTRIGHT(x, 19) ^ (x >> 10))

#define BLOCK_SIZE 256

__constant__ uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

__device__ void sha256_transform(uint32_t *state, const unsigned char data[])
{
    uint32_t a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

    for (i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
    for (; i < 64; ++i)
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];

    for (i = 0; i < 64; ++i)
    {
        t1 = h + EP1(e) + CH(e, f, g) + k[i] + m[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

__device__ void sha256_init(uint32_t *state)
{
    state[0] = 0x6a09e667;
    state[1] = 0xbb67ae85;
    state[2] = 0x3c6ef372;
    state[3] = 0xa54ff53a;
    state[4] = 0x510e527f;
    state[5] = 0x9b05688c;
    state[6] = 0x1f83d9ab;
    state[7] = 0x5be0cd19;
}

__device__ void sha256_update(uint32_t *state, const unsigned char *data, size_t len, unsigned char *buffer, size_t *bitlen)
{
    size_t i;

    for (i = 0; i < len; ++i)
    {
        buffer[*bitlen >> 3] = data[i];
        *bitlen += 8;
        if (*bitlen == 512)
        {
            sha256_transform(state, buffer);
            *bitlen = 0;
        }
    }
}

__device__ void sha256_final(uint32_t *state, unsigned char *buffer, size_t *bitlen, unsigned char *hash)
{
    size_t i;

    i = *bitlen >> 3;
    buffer[i++] = 0x80;
    while (i < 56)
        buffer[i++] = 0x00;

    for (int j = 56; j < 64; ++j)
    {
        buffer[j] = 0;
    }

    sha256_transform(state, buffer);

    for (i = 0; i < 4; ++i)
    {
        hash[i] = (state[0] >> (24 - i * 8)) & 0xff;
        hash[i + 4] = (state[1] >> (24 - i * 8)) & 0xff;
        hash[i + 8] = (state[2] >> (24 - i * 8)) & 0xff;
        hash[i + 12] = (state[3] >> (24 - i * 8)) & 0xff;
        hash[i + 16] = (state[4] >> (24 - i * 8)) & 0xff;
        hash[i + 20] = (state[5] >> (24 - i * 8)) & 0xff;
        hash[i + 24] = (state[6] >> (24 - i * 8)) & 0xff;
        hash[i + 28] = (state[7] >> (24 - i * 8)) & 0xff;
    }
}

// CUDA kernel to compute the hash
__global__ void mine_kernel(const char *block_data, int block_data_len, int difficulty, unsigned char *valid_hash, int *nonce_found, int max_nonce)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_nonce)
        return;

    // Initialize SHA-256 state
    uint32_t state[8];
    sha256_init(state);

    // Copy the block data and append the nonce
    unsigned char buffer[64] = {0};
    size_t bitlen = 0;
    sha256_update(state, (unsigned char *)block_data, block_data_len, buffer, &bitlen);
    sha256_update(state, (unsigned char *)&idx, sizeof(idx), buffer, &bitlen);

    // Finalize the hash computation
    unsigned char hash[32];
    sha256_final(state, buffer, &bitlen, hash);

    // Check if the hash is valid by looking for leading zeros
    bool valid = true;
    for (int i = 0; i < difficulty; ++i)
    {
        if (hash[i] != 0)
        {
            valid = false;
            break;
        }
    }

    // If the hash is valid, store the result and the nonce
    if (valid && atomicCAS(nonce_found, -1, idx) == -1)
    {
        memcpy(valid_hash, hash, 32);
    }
}

// Host function to launch the kernel and handle the mining
extern "C" void mine_block(const char *block_data, int block_data_len, int difficulty, unsigned char *valid_hash, int *nonce_found, int max_nonce)
{
    char *d_block_data;
    unsigned char *d_valid_hash;
    int *d_nonce_found;

    cudaMalloc(&d_block_data, block_data_len);
    cudaMalloc(&d_valid_hash, 32 * sizeof(unsigned char));
    cudaMalloc(&d_nonce_found, sizeof(int));

    cudaMemcpy(d_block_data, block_data, block_data_len, cudaMemcpyHostToDevice);
    cudaMemset(d_nonce_found, -1, sizeof(int));

    int num_blocks = (max_nonce + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mine_kernel<<<num_blocks, BLOCK_SIZE>>>(d_block_data, block_data_len, difficulty, d_valid_hash, d_nonce_found, max_nonce);

    cudaMemcpy(valid_hash, d_valid_hash, 32 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(nonce_found, d_nonce_found, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_block_data);
    cudaFree(d_valid_hash);
    cudaFree(d_nonce_found);
}
