#include <vector>
#include <cmath>

const int N = 64;//threads * blocks;

__global__ void kernel(float *x, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

int main(){
    std::vector<float> data_vector(N);

    const int num_streams = 8;
    cudaStream_t streams[num_streams];
    float *data[num_streams];

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]); // crea uno degli 8 stream
        cudaMalloc(&data[i], N * sizeof(float));

        kernel<<<1, 64, 0, streams[i]>>>(data[i], N); // lancia un kernel per stream 
        kernel<<<1, 1>>>(0, 0); // lancia un “dummy kernel” nel default stream
    }

    cudaDeviceReset();

    return 0;
}