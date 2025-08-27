#ifndef ED25519_H
#define ED25519_H

#include <stddef.h>
#define ED25519_DECLSPEC

// Create batches of seeds and key pairs
__host__ void ED25519_DECLSPEC ed25519_kernel_create_seed(unsigned char *seed, int batch_size);
__global__ void ED25519_DECLSPEC ed25519_kernel_create_keypair_batch(unsigned char *public_key, unsigned char *private_key, const unsigned char *seed, int limit);

// Single key-pair batch functions
__global__ void ED25519_DECLSPEC ed25519_kernel_sign_batch_single_keypair(unsigned char *signature, const unsigned char *message, size_t *message_len, const unsigned char *public_key, const unsigned char *private_key, int limit);
__global__ void ED25519_DECLSPEC ed25519_kernel_verify_batch_single_keypair(const unsigned char *signature, const unsigned char *message, size_t *message_len, const unsigned char *public_key, int *verified, int limit);

// Multiple key-pair batch functions
__global__ void ED25519_DECLSPEC ed25519_kernel_sign_batch_multi_keypair(unsigned char *signature, const unsigned char *message, size_t *message_len, const unsigned char *public_key, const unsigned char *private_key, int *key_mapping, int limit);
__global__ void ED25519_DECLSPEC ed25519_kernel_verify_batch_multi_keypair(const unsigned char *signature, const unsigned char *message, size_t *message_len, const unsigned char *public_key, int *verified, int *key_mapping, int limit);

// Add scalar batch functions
__global__ void ED25519_DECLSPEC ed25519_kernel_add_scalar_batch(unsigned char *public_key, unsigned char *private_key, const unsigned char *scalar, int limit);
__global__ void ED25519_DECLSPEC ed25519_kernel_add_multi_scalar_batch(unsigned char *public_key, unsigned char *private_key, const unsigned char *scalar, int *scalar_mapping, int limit);

// Generate shared secret for a batch of key pairs
__global__ void ED25519_DECLSPEC ed25519_kernel_key_exchange_batch(unsigned char *shared_secret, const unsigned char *public_key, const unsigned char *private_key, int limit);

#endif
