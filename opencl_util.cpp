#include "opencl_util.hpp"




void print_trace(void) {
    char **strings;
    size_t i, size;
    enum Constexpr { MAX_SIZE = 1024 };
    void *array[MAX_SIZE];
    size = backtrace(array, MAX_SIZE);
    strings = backtrace_symbols(array, size);
    for (i = 0; i < size; i++)
        printf("%s\n", strings[i]);
    puts("");
    free(strings);
}


OpenCLHandler::OpenCLHandler() {

    cl_uint retNumDevices, retNumPlatforms;
    OPENCL_CHECK(clGetPlatformIDs(1, &platformId, &retNumPlatforms));
    OPENCL_CHECK(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &retNumDevices));

    context = clCreateContext(nullptr, 1, &deviceID, nullptr, nullptr,  &ret); OPENCL_CHECK(ret);
    cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    command_queue = clCreateCommandQueueWithProperties(context, deviceID, properties, &ret); OPENCL_CHECK(ret);
    // _program = clCreateProgramWithSource(context, 1, &kernel_source, &kernel_size, &ret); OPENCL_CHECK(ret);
    // OPENCL_BUILD_CHECK(clBuildProgram(_program, 1, &_deviceID, "-cl-std=CL2.0", nullptr, nullptr));

}


cl_kernel OpenCLHandler::get_kernel(std::string kernel_name) {
    
    std::string kernel_dir = "../kernels/";
        if (kernel_name.find("ckpt") != std::string::npos) {
        std::ifstream kernelFile;
        kernelFile.open(kernel_dir+kernel_name+".so", std::ios::binary | std::ios::ate);
        size_t kernelSize = kernelFile.tellg();
        kernelFile.seekg(0, std::ios::beg);
        std::vector<unsigned char> kernelSource(kernelSize);
        kernelFile.read(reinterpret_cast<char *>(kernelSource.data()), kernelSize);
        auto kernel_str = kernelSource.data();

        cl_int binary_status;
        cl_program program = clCreateProgramWithBinary (context, 1, &deviceID, (const size_t *)&kernelSize, (const unsigned char **)&kernel_str, &binary_status, &ret);
        OPENCL_CHECK(ret); OPENCL_CHECK(binary_status);

        // Build program
        OPENCL_CHECK(clBuildProgram(program, 1, &deviceID, nullptr, nullptr, nullptr));

        cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &ret);
        OPENCL_CHECK(ret);

        return kernel;
    } else {
        std::ifstream kernel_file(kernel_dir + kernel_name + ".cl");
        std::string kernel_code((std::istreambuf_iterator<char>(kernel_file)), std::istreambuf_iterator<char>());

        auto kernel_source = kernel_code.c_str();
        auto kernel_size = kernel_code.size();

        auto program = clCreateProgramWithSource(context, 1, &kernel_source, &kernel_size, &ret); OPENCL_CHECK(ret);

        OPENCL_BUILD_CHECK(clBuildProgram(program, 1, &deviceID, "-cl-std=CL2.0", nullptr, nullptr));

        cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &ret); OPENCL_CHECK(ret);
        OPENCL_CHECK(ret);
        return kernel;
    }



}


cl_kernel OpenCLHandler::get_kernel_binary(std::string kernel_name) {
    
    std::string routine_dir = "../routines/";
    
    std::ifstream kernelFile;
    kernelFile.open(routine_dir+kernel_name, std::ios::binary | std::ios::ate);
    size_t kernelSize = kernelFile.tellg();
    kernelFile.seekg(0, std::ios::beg);
    std::vector<unsigned char> kernelSource(kernelSize);
    kernelFile.read(reinterpret_cast<char *>(kernelSource.data()), kernelSize);
    auto kernel_str = kernelSource.data();

    cl_int binary_status;
    cl_program program = clCreateProgramWithBinary (context, 1, &deviceID, (const size_t *)&kernelSize, (const unsigned char **)&kernel_str, &binary_status, &ret);
    OPENCL_CHECK(ret); OPENCL_CHECK(binary_status);

    // Build program
    OPENCL_CHECK(clBuildProgram(program, 1, &deviceID, nullptr, nullptr, nullptr));

    cl_kernel kernel = clCreateKernel(program, "routine", &ret);
    OPENCL_CHECK(ret);

    return kernel;
}

OpenCLHandler::~OpenCLHandler() {
    // OPENCL_CHECK(clFlush(_command_queue));
    // OPENCL_CHECK(clFinish(_command_queue));
    // OPENCL_CHECK(clReleaseCommandQueue(_command_queue));

    // for (auto &x:_buffers) {
    // 	OPENCL_CHECK(clReleaseMemObject(x));
    // }

    // OPENCL_CHECK(clReleaseProgram(_program));
    OPENCL_CHECK(clReleaseContext(context));
}


void OpenCLHandler::launch_kernel_timer(
    cl_kernel kernel,
    std::vector<std::pair<size_t, std::shared_ptr<void *>>> const& arg_list,
    std::vector<size_t> const& global_size,
    std::vector<size_t> const& local_size,
    cl_event* event,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list
) {

    for (size_t i = 0; i < arg_list.size(); ++i) {
        OPENCL_CHECK(clSetKernelArg(kernel, i, arg_list[i].first, arg_list[i].second.get()));
    }

    OPENCL_CHECK(
        clEnqueueNDRangeKernel(
            command_queue,
            kernel,
            global_size.size(),
            nullptr,
            global_size.data(),
            local_size.data(),
            num_events_in_wait_list,
            event_wait_list,
            event
        )
    );
}


