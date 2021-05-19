__kernel void vecadd_pt_measure(__global float* restrict a, 
	__global float* restrict b,
	__global float* restrict c) {

	for(int measure_iter = 0; measure_iter < 1024*1024; ++measure_iter) { //when measure the PC offset, we set a long exec time
		int gid = get_global_id(0);
        #pragma unroll
		for(int i = 0; i < 16; ++i) {
			c[gid+1024*256*i] = a[gid+1024*256*i] + b[gid+1024*256*i];
		}
	}
}