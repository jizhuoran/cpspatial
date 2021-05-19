#include <map>
#include <unistd.h>

#include "job_atom.hpp"
#include "opencl_util.hpp"
#include <amd-dbgapi.h>

extern std::map<std::string, KernelInfo*> kernel_global;
extern std::map<std::string, KernelInfo*> routine_info_global;
extern OpenCLHandler handler;

JobAtom::JobAtom(
    std::string _kernel_name,
    JobType _job_type) 
: kernel_name(_kernel_name), 
job_type(_job_type) {
    if(_job_type != JobType::ROUTINE) {
        kernel_info = kernel_global[_kernel_name];
    } else {
        kernel_info = routine_info_global[_kernel_name+".routine"];
        kernel_name = "routine"; //we set all routine's kernal name is routine
    }
}

int JobAtom::launch_to_get_time() {

    // cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};

    // cl_int ret;
    // command_queue = clCreateCommandQueueWithProperties(handler.context, handler.deviceID, properties, &ret); OPENCL_CHECK(ret);
    
    float total_ms = .0;

    if(job_type==JobType::ROUTINE) {
        cl_ulong time_start;
        cl_ulong time_end;
        for(int j = 0; j < kernel_info->num_warp; ++j) {
            cl_event event;
            handler.launch_kernel_timer(
                kernel_info->kernel,
                kernel_info->arg_list, kernel_info->global_size, kernel_info->local_size,
                &event, 0, nullptr
            );
            OPENCL_CHECK(clWaitForEvents(1, &event));
            OPENCL_CHECK(clFinish(handler.command_queue));
            clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
            clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
            total_ms += float(time_end-time_start)/1e3;
        }
    } else {
        for(int i = 0; i < WARPCU; ++i) { 
            cl_ulong time_start;
            cl_ulong time_end;
            for(int j = 0; j < kernel_info->num_warp; ++j) {
                cl_event event;
                handler.launch_kernel_timer(
                    kernel_info->kernel,
                    kernel_info->arg_list, kernel_info->global_size, kernel_info->local_size,
                    &event, 0, nullptr
                );
                OPENCL_CHECK(clWaitForEvents(1, &event));
                OPENCL_CHECK(clFinish(handler.command_queue));
                clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
                clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
                total_ms += float(time_end-time_start)/1e3;
            }
        }
    }


    needed_exec_time = total_ms * TIMESTAMP_GRAIN;

    exec_time = needed_exec_time;
    return exec_time;
    // has_launched_ = true;
}


int JobAtom::get_offset() {
    int offset = 0;
    cl_event event;

    auto meauser_kernel_info = kernel_global[kernel_info->kernel_name+"_measure"];

    handler.launch_kernel_timer(
            meauser_kernel_info->kernel,
            meauser_kernel_info->arg_list, meauser_kernel_info->global_size, meauser_kernel_info->local_size,
            &event, 0, nullptr
    );
    usleep(1000); //make sure the kernel is successful started
    offset = amd_dbgapi_wave_get_pc_offset(getpid(), meauser_kernel_info->kernel_name);
    OPENCL_CHECK(clWaitForEvents(1, &event));
    OPENCL_CHECK(clFinish(handler.command_queue));

    return offset;
}


// bool JobAtom::has_finished() {
//     if(has_finished_) return true;
//     if(!has_launched_) return false;
//     cl_int status;
//     OPENCL_CHECK(clGetEventInfo(
//         event,
//         CL_EVENT_COMMAND_EXECUTION_STATUS,
//         sizeof(status),
//         &status,
//         nullptr)
//     );
//     return status;
// }

// float JobAtom::get_execute_time() {
//     if(!has_finished_) { 
//         return -1.; 
//     } else {
//         clWaitForEvents(1, &event);
//         clFinish(command_queue);
//         cl_ulong time_start;
//         cl_ulong time_end;
//         clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
//         clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
//         return time_end-time_start;
//     }
// }