__global__ void kernel_add_scalar_copy_scalar(unsigned char *a, const unsigned char *b) {
  int i = threadIdx.x;
  a[i] = b[i];
}

__global__ void kernel_add_scalar_copy_pk(unsigned char *pk, unsigned char *b) {
  int i = threadIdx.x;
  pk[i + 32] = b[i];
}
