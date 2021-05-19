#include <map>
#include <iterator>
#include <fstream>
#include <string>
#include <sstream>

#include "kernel_info.hpp"
#include "opencl_util.hpp"

extern OpenCLHandler handler;

class line : public std::string {};

std::istream &operator>>(std::istream &is, line &l)
{
    std::getline(is, l);
    return is;
}

std::vector<std::string> split(std::string const &input) { 
    std::istringstream buffer(input);
    std::vector<std::string> ret;

    std::copy(std::istream_iterator<std::string>(buffer), 
              std::istream_iterator<std::string>(),
              std::back_inserter(ret));
    return ret;
}

KernelInfo::KernelInfo(
    std::string _kernel_name,
    std::vector<size_t> _global_size,
    std::vector<size_t> _local_size,
    std::vector<std::pair<size_t, std::shared_ptr<void*>>> _arg_list, 
    JobType _job_type, int _num_warp) //here we just use BATCH_JOB to distinguish ROUTINE
: kernel_name(_kernel_name), 
global_size(_global_size), local_size(_local_size),
arg_list(_arg_list), num_warp(_num_warp) {
    if(_job_type == JobType::ROUTINE) {
        kernel = handler.get_kernel_binary(_kernel_name); //for routine, we read binary
    } else {
        kernel = handler.get_kernel(_kernel_name);
    }

}

void KernelInfo::launch(cl_event* event) {
    handler.launch_kernel_timer(
        kernel,
        arg_list, global_size, local_size,
        event, 0, nullptr
    );
}

std::map<std::string, KernelInfo*> construct_kernel_infos() {

    std::map<std::string, KernelInfo*> ret;

    std::ifstream kernel_file("../kernel_info.config");

    std::istream_iterator<line> begin(kernel_file);
    std::istream_iterator<line> end;

    for(std::istream_iterator<line> it = begin; it != end; ++it) {
        auto kernel_info = split(std::string(*it));
        ret[kernel_info[0]] = new KernelInfo(
            kernel_info[0],
            std::vector<size_t>{static_cast<size_t>(std::stol(kernel_info[1]))},
            std::vector<size_t>{static_cast<size_t>(std::stol(kernel_info[2]))},
            {
                std::make_pair<size_t, std::shared_ptr<void*>>(sizeof(cl_mem), handler.create_buffer<float>(std::stol(kernel_info[3]))),
                std::make_pair<size_t, std::shared_ptr<void*>>(sizeof(cl_mem), handler.create_buffer<float>(std::stol(kernel_info[3]))),
                std::make_pair<size_t, std::shared_ptr<void*>>(sizeof(cl_mem), handler.create_buffer<float>(std::stol(kernel_info[3])))
            }
        );
    }
    return ret;
}

std::map<std::string, KernelInfo*> kernel_global = construct_kernel_infos();