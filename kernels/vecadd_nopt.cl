__kernel void vecadd_nopt(__global float * restrict a, 
	__global float * restrict b,
	__global float * restrict c) {
		int gid = get_global_id(0);
		c[gid] = a[gid] + b[gid];
}