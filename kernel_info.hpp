#pragma once

#include <string>
#include <vector>
#include <memory>
#include "opencl_util.hpp"

#define WARPCU 64
#define COUNTING_THEAD 0.2

enum class JobType {
    BATCH_JOB = 0,
    LS_JOB = 1,
    ROUTINE = 2
};

class KernelInfo {
public:
    KernelInfo(
        std::string _kernel_name,
        std::vector<size_t> _global_size,
	    std::vector<size_t> _local_size,
        std::vector<std::pair<size_t, std::shared_ptr<void*>>> _arg_list, 
        JobType _job_type = JobType::BATCH_JOB, int _num_warp = 1); //here we just use BATCH_JOB to distinguish ROUTINE

    void launch(cl_event* event);

    std::string kernel_name;
    std::vector<size_t> global_size;
	std::vector<size_t> local_size;
    std::vector<std::pair<size_t, std::shared_ptr<void*>>> arg_list;

    cl_kernel kernel;

    int num_warp = 1;

private:
	KernelInfo(const KernelInfo&);
  	KernelInfo& operator=(const KernelInfo&);
};