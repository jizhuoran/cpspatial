__kernel void vecadd_nopt_measure(__global float * a, 
	__global float * b,
	__global float * c) {
	
	for(int measure_iter = 0; measure_iter < 1024*1024; ++measure_iter) { //when measure the PC offset, we set a long exec time
		int gid = get_global_id(0);
		c[gid] = a[gid] + b[gid];
	}

}