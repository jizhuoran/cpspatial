#pragma once

#include <iostream>
#include <fstream>
#include <streambuf>
#include <vector>
#include <memory>
#include <filesystem>

#define CL_TARGET_OPENCL_VERSION 220
#define TIMESTAMP_GRAIN 30

#include <CL/cl.h>
// #include "cl_error_code.inc"


#include <stdio.h>
#include <execinfo.h>
void print_trace(void);

#define OPENCL_CHECK(condition) \
  do { \
    cl_int error = condition; \
    if(error != CL_SUCCESS) { \
      std::cerr << "This is a error for OpenCL "<< get_error_string(error) << " in " << __LINE__ << " in " << __FILE__ << std::endl;\
	  print_trace(); \
      exit(1); \
    } \
  } while (0)


#define OPENCL_BUILD_CHECK(condition) \
  do { \
    cl_int error = condition; \
    char *buff_erro; \
    cl_int errcode; \
    size_t build_log_len; \
    errcode = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, 0, nullptr, &build_log_len); \
	if (errcode) { \
    	std::cerr << "clGetProgramBuildInfo failed at line " << __LINE__ << std::endl; \
    	exit(-1); \
    } \
  	buff_erro = (char *)malloc(build_log_len); \
  	if (!buff_erro) { \
      	std::cerr << "malloc failed at line" << __LINE__ << std::endl; \
      	exit(-2); \
  	} \
    errcode = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, build_log_len, buff_erro, nullptr); \
    if (errcode) { \
        std::cerr << "clGetProgramBuildInfo failed at line " << __LINE__ << std::endl; \
        exit(-3); \
    } \
    free(buff_erro); \
    if (error != CL_SUCCESS) { \
      std::cerr << "clBuildProgram failed" << std::endl; \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

    // std::cerr << "Build log: " << buff_erro << std::endl; \

const char *get_error_string(cl_int error);

class OpenCLHandler
{
public:

	OpenCLHandler();

    cl_kernel get_kernel(std::string kernel_name);

	cl_kernel get_kernel_binary(std::string kernel_name);


	// OpenCLHandler(std::string kernel_path, size_t global_size) {

	// 	std::ifstream kernel_file(kernel_path);
    // 	std::string kernel_code((std::istreambuf_iterator<char>(kernel_file)), std::istreambuf_iterator<char>());
    // 	// kernel_code = fmt::format("#define SIZE ({})\n#define ONLINE_BUILD 1\n\n", global_size) + kernel_code;

	// 	cl_uint retNumDevices, retNumPlatforms;
	// 	OPENCL_CHECK(clGetPlatformIDs(1, &_platformId, &retNumPlatforms));
	// 	OPENCL_CHECK(clGetDeviceIDs(_platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &retNumDevices));

	// 	context = clCreateContext(nullptr, 1, &deviceID, nullptr, nullptr,  &ret); OPENCL_CHECK(ret);

	// 	cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
	// 	_command_queue = clCreateCommandQueueWithProperties(context, deviceID, properties, &ret); OPENCL_CHECK(ret);

	// 	auto kernel_source = kernel_code.c_str();
	// 	auto kernel_size = kernel_code.size();

	// 	_program = clCreateProgramWithSource(context, 1, &kernel_source, &kernel_size, &ret); OPENCL_CHECK(ret);

	// 	OPENCL_BUILD_CHECK(clBuildProgram(_program, 1, &_deviceID, "-cl-std=CL2.0", nullptr, nullptr));

	// }

	~OpenCLHandler();


	void launch_kernel_timer(
	  cl_kernel kernel,
	  std::vector<std::pair<size_t, std::shared_ptr<void *>>> const& arg_list,
	  std::vector<size_t> const& global_size,
	  std::vector<size_t> const& local_size,
	  cl_event* event,
	  cl_uint num_events_in_wait_list,
	  const cl_event *event_wait_list
	);

	template <typename Dtype>
	std::shared_ptr<void*> create_buffer(size_t count, cl_mem_flags flags = CL_MEM_READ_WRITE) {

        auto buffer = std::make_shared<void*>();
        *buffer = clCreateBuffer(context, flags, sizeof(Dtype) * count, nullptr, &ret); OPENCL_CHECK(ret);
		return buffer;
	}

	// template <typename Dtype>
	// cl_mem create_buffer_from_vector(const std::vector<Dtype> &host_buffer, cl_mem_flags flags = CL_MEM_READ_WRITE) {
	// 	cl_mem buffer = clCreateBuffer(context, flags, sizeof(Dtype) * host_buffer.size(), nullptr, &ret); OPENCL_CHECK(ret);
	// 	OPENCL_CHECK(clEnqueueWriteBuffer(_command_queue, buffer, CL_TRUE, 0, sizeof(Dtype) * host_buffer.size(), host_buffer.data(), 0, nullptr, nullptr));
		// _buffers.push_back(buffer);
	// 	return buffer;
	// }

	// template <typename Dtype>
	// void write_buffer(const std::vector<Dtype> &host_buffer, cl_mem device_buffer) {
	// 	OPENCL_CHECK(clEnqueueWriteBuffer(_command_queue, device_buffer, CL_TRUE, 0, sizeof(Dtype) * host_buffer.size(), host_buffer.data(), 0, nullptr, nullptr));
	// }

	// template <typename Dtype>
	// void read_buffer(std::vector<Dtype> &host_buffer, cl_mem device_buffer) {
	// 	OPENCL_CHECK(clEnqueueReadBuffer(_command_queue, device_buffer, CL_TRUE, 0, sizeof(Dtype) * host_buffer.size(), host_buffer.data(), 0, nullptr, nullptr));
	// }



    cl_platform_id platformId = nullptr;
	cl_device_id deviceID = nullptr;

	cl_context context;
	cl_command_queue command_queue;
	// std::map<std::string, KernelInfo*> routine_info_global;

private:

	std::vector<std::shared_ptr<cl_mem>> _buffers;

	cl_int ret;


	// cl_command_queue _command_queue;
	// cl_program _program;

	OpenCLHandler(const OpenCLHandler&);
  	OpenCLHandler& operator=(const OpenCLHandler&);

};