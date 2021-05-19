__kernel void vecadd_LS(__global float * restrict a, 
	__global float * restrict b,
	__global float * restrict c) {
	int gid = get_global_id(0);
	#pragma unroll
	for(int i = 0; i < 16; ++i) {
		c[gid+1024*8*i] = a[gid+1024*8*i] + b[gid+1024*8*i];
	}
}