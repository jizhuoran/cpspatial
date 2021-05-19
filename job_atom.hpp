#pragma once

#include "opencl_util.hpp"
#include "kernel_info.hpp"

class JobAtom {

public:
    JobAtom(
        std::string _kernel_name,
        JobType _job_type);

    JobAtom(int _exec_time) : exec_time(_exec_time) {} //wait queue
    
    std::string kernel_name;
    JobType job_type;
    KernelInfo* kernel_info;

    // std::chrono::time_point<std::chrono::system_clock> submit_time;
    // std::chrono::time_point<std::chrono::system_clock> launch_time;
    // std::chrono::time_point<std::chrono::system_clock> finish_time;


    uint64_t offset;

    bool has_launched_ = false;
    bool has_finished_ = false;
    

    int launch_to_get_time();

    int exec_time;
    int needed_exec_time;
    int finish_time;
    float init_progress = .0;
    // bool has_finished();

    // float get_execute_time();

    void busy_check_job() {
        exec_time--;
    }

    // int submit_time() {
    //     return submit_time;
    // }

    // int exec_time() {
    //     return exec_time;
    // }

    bool has_finished() {
        return exec_time <= 0;
    }

    int get_offset();

    float useful_work_has_done() { 
        return 1 - init_progress;
    }

    void increase_init_progress(float progress) {
        init_progress += progress;
    }

private:
	JobAtom(const JobAtom&);
  	JobAtom& operator=(const JobAtom&);
};