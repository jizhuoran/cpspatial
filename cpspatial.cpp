#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <memory>
#include <chrono>
#include <random>
#include <ctime>
#include <amd-dbgapi.h>


#include "opencl_util.hpp"
#include "kernel_info.hpp"
#include "job_atom.hpp"

OpenCLHandler handler;

extern std::map<std::string, KernelInfo*> kernel_global;
std::map<std::string, KernelInfo*> routine_info_global;

float compute_throughput(std::string test_case, 
    int exec_queue_id, int preempt_queue_id, 
    std::shared_ptr<JobAtom> job_atom, bool need_resume = true) {
    
    if(need_resume) {
        if(preempt_queue_id == 0) {
            return 0;
        } else {
            return float(job_atom->needed_exec_time - job_atom->exec_time) / job_atom->needed_exec_time;
        }
    } else {
        float progress =  float(job_atom->needed_exec_time - job_atom->exec_time) / job_atom->needed_exec_time;
        return progress < COUNTING_THEAD ? .0 : progress; 
    }        
}

class AtomQueue {
public:

    AtomQueue(JobType _job_type) : job_type(_job_type) { }

    void enqueue(std::shared_ptr<JobAtom> job_atom) {
        atom_queue.push_back(job_atom);
    }
    
    virtual std::pair<int, int> busy_check_queue() {
        int free_SM = 0;
        for(auto& job : atom_queue) {
            job->busy_check_job();
            if(job->has_finished()) {
                free_SM += 1;
            }
        }
        atom_queue.erase(
            std::remove_if(
                atom_queue.begin(),
                atom_queue.end(),
                [&](auto& x) -> bool { return x->has_finished(); }
            ), 
            atom_queue.end()
        );
        return {free_SM, 0};
    }

    bool empty() { return atom_queue.empty(); }
    size_t num_atom() { return atom_queue.size(); }
    
    virtual void dequeue(int preemp_index = 0) {
        auto job_atom = atom_queue[preemp_index];
        atom_queue.erase(atom_queue.begin() + preemp_index);
        // return job_atom;
    }

    JobType job_type;
    std::vector<std::shared_ptr<JobAtom>> atom_queue;
};

class ExecuteQueue : public AtomQueue {

public:
    ExecuteQueue(std::string _kernel_name, JobType _job_type)
    : AtomQueue(_job_type), kernel_name(_kernel_name) { }

    std::shared_ptr<JobAtom> submit_kernel() {
        auto job = std::make_shared<JobAtom>(
            kernel_name, job_type);
        atom_queue.push_back(job);
        return job;
    }

    std::shared_ptr<JobAtom> submit_and_run_kernel() {
        auto job = submit_kernel();
        job->launch_to_get_time();
        return job;
    }

    virtual std::pair<int, int> busy_check_queue() {
        int free_SM = 0, finished_job = 0;
        for(auto& job : atom_queue) {
            job->busy_check_job();
            if(job->has_finished()) {
                free_SM += 1;
                finished_job += job->useful_work_has_done();
            }
        }
        atom_queue.erase(
            std::remove_if(
                atom_queue.begin(),
                atom_queue.end(),
                [&](auto& x) -> bool { return x->has_finished(); }
            ), 
            atom_queue.end()
        );
        return {free_SM, finished_job};
    }

    virtual void dequeue(int preemp_index = 0) {
        throw;
    }

    std::string kernel_name;

};

    // virtual std::shared_ptr<JobAtom> dequeue(int preemp_index = 0) {
    //     throw; return {};
    // }

std::string get_preempt_routine(std::string kernel_name, int preept_mech, int offset) {
    std::stringstream ss;
    ss << kernel_name << "_preempt_" << preept_mech << "_" << offset;
    std::string ret = ss.str();
    std::cout << ret << std::endl;
    return ret;
}

std::string get_resume_routine(std::string kernel_name, int preept_mech, int offset) {
    std::stringstream ss;
    ss << kernel_name << "_resume" << preept_mech << "_" << offset; 
    //TODO: here _resume should be _resume_, we need to correct the generator first and then corret this
    std::string ret = ss.str();
    return ret;
}


class BatchQueue : public ExecuteQueue {

public:
    BatchQueue(std::string _kernel_name,
        int _eq_id, int _pq_id,
        int _supposed_size = -1, int _low_threshold = -1)
    : ExecuteQueue(_kernel_name, JobType::BATCH_JOB), 
    eq_id(_eq_id), pq_id(_pq_id),
    supposed_size(_supposed_size), low_threshold(_low_threshold) { 

        preempt_queue_ = std::make_unique<AtomQueue>(JobType::ROUTINE);
        resume_queue_  = std::make_unique<AtomQueue>(JobType::ROUTINE);

    }


    int normthrough() {
        if (this->kernel_name.find("nopt") != std::string::npos) {
            return 1; //for nopt kernel, each kernel only compute 1/16 of the pt kernel
        } else {
            return 16;
        }
    }


    virtual void dequeue(int preemp_index = 0) override {
        auto job_atom = atom_queue[preemp_index];
        atom_queue.erase(atom_queue.begin() + preemp_index);
        //TODO launch preempt routine

        job_atom->offset = job_atom->get_offset();
        std::shared_ptr<JobAtom> preempt_routine;

        switch (pq_id) {
            case 0:
                preempt_routine = std::make_shared<JobAtom>(1); //launch a fake kernel
                preempt_queue_->enqueue(preempt_routine);
                break;
            case 1:
                preempt_routine = std::make_shared<JobAtom>(
                    get_preempt_routine(kernel_name, 1, job_atom->offset),
                    JobType::ROUTINE);
                preempt_queue_->enqueue(preempt_routine);
                preempt_routine->launch_to_get_time();
                to_be_resume.push_back(job_atom);
                break;
            case 2:
                preempt_routine = std::make_shared<JobAtom>(
                    get_preempt_routine(kernel_name, 2, job_atom->offset),
                    JobType::ROUTINE);
                preempt_queue_->enqueue(preempt_routine);
                preempt_routine->launch_to_get_time();
                to_be_resume.push_back(job_atom);
                break;
            default:
                throw;
        }
    }

    bool try_resume() {
        if(to_be_resume.size() == 0) return false;
        auto job_atom = to_be_resume.back();
        to_be_resume.pop_back();
        std::shared_ptr<JobAtom> resume_routine;
        //TODO launch resume routine
        switch (pq_id) {
            case 0:
                throw;
                break;
            case 1:
                resume_routine = std::make_shared<JobAtom>(
                    get_resume_routine(kernel_name, 1, job_atom->offset),
                    JobType::ROUTINE);
                resume_routine->launch_to_get_time();
                job_atom->exec_time += resume_routine->needed_exec_time;
                atom_queue.push_back(job_atom);
                break;
            case 2:
                resume_routine = std::make_shared<JobAtom>(
                    get_resume_routine(kernel_name, 2, job_atom->offset),
                    JobType::ROUTINE);
                resume_routine->launch_to_get_time();
                job_atom->exec_time += resume_routine->needed_exec_time;
                atom_queue.push_back(job_atom);
                break;
            default:
                throw;
        }
        return true;
    }

    virtual std::pair<int, int> busy_check_queue() override {
        auto general_info = ExecuteQueue::busy_check_queue();
        int finished_job = general_info.second;
        int free_SM = general_info.first;
        free_SM += preempt_queue_->busy_check_queue().first;
        free_SM += resume_queue_->busy_check_queue().first;
        return {free_SM, finished_job};
    }


    bool full() {
        return atom_queue.size() == supposed_size; 
    }

    float stop() {
        float progress_sum = .0;
        for(const auto& job_atom:this->atom_queue) {
            progress_sum += compute_throughput(kernel_name, eq_id, pq_id, job_atom, false);
        }
        for(auto& job_atom: to_be_resume) {
            std::shared_ptr<JobAtom> resume_routine;
            std::shared_ptr<JobAtom> preemt_routine;
            switch (pq_id) {
            case 0:
                break;
                throw;
            case 1:
                resume_routine = std::make_shared<JobAtom>(
                    get_resume_routine(kernel_name, 1, job_atom->offset),
                    JobType::ROUTINE);
                resume_queue_->enqueue(resume_routine);
                resume_routine->launch_to_get_time();
                preemt_routine = std::make_shared<JobAtom>(
                    get_preempt_routine(kernel_name, 1, job_atom->offset),
                    JobType::ROUTINE);
                resume_queue_->enqueue(preemt_routine);
                preemt_routine->launch_to_get_time();
                break;
            case 2:
                resume_routine = std::make_shared<JobAtom>(
                    get_resume_routine(kernel_name, 2, job_atom->offset),
                    JobType::ROUTINE);
                resume_queue_->enqueue(resume_routine);
                resume_routine->launch_to_get_time();
                preemt_routine = std::make_shared<JobAtom>(
                    get_preempt_routine(kernel_name, 2, job_atom->offset),
                    JobType::ROUTINE);
                resume_queue_->enqueue(preemt_routine);
                preemt_routine->launch_to_get_time();
                job_atom->exec_time -= preemt_routine->needed_exec_time;
                //for CS-Defer, the inst executed before CS contributes
                break;
            default:
                throw;
            }
            job_atom->exec_time += resume_routine->needed_exec_time;
            progress_sum += compute_throughput(kernel_name, eq_id, pq_id, job_atom, true);
        }
        return progress_sum;
    }

    std::vector<std::shared_ptr<JobAtom>> to_be_resume;


    std::unique_ptr<AtomQueue> preempt_queue_;
    std::unique_ptr<AtomQueue> resume_queue_;



    //for CPSpatial only
    int supposed_size;
    int low_threshold;
    int pq_id;
    int eq_id;
    int onfly = 0;

    // float acc_latency = .0;
    // float acc_overhead = .0;

};



class ScheduleSystem {
public:
    ScheduleSystem(std::string _batch_name, std::string _LS_name) 
    : batch_name(_batch_name), LS_name(_LS_name) {
        execute_LS_queue = std::make_shared<ExecuteQueue>(_LS_name, JobType::LS_JOB);
    }

    std::string batch_name;
    std::string LS_name;
    std::vector<std::shared_ptr<BatchQueue>> batch_queues;

    // RoutineQueue under_preemption_queue();

    std::shared_ptr<ExecuteQueue> execute_LS_queue;
    std::vector<int> wait_LS_queue;
    
    int cur_timestamp = 0;
    int test_period = 0;
    int freeSM = 0;
    int served_SM = 0;
    float finished_LS = 0;

    int acc_latency = 0;
    int acc_throughput = 0;

    std::vector<std::pair<int, int>> ls_occupy;

    void submit_LS() {
        wait_LS_queue.push_back(cur_timestamp);
        for(auto & batch_queue : batch_queues) {
            if (!batch_queue->empty()) {
                batch_queue->dequeue();
                return;
            } else {
                if (batch_queue->onfly > 0) {
                    batch_queue->onfly--;
                }
            }
        }
    }

    void launch_waiting_LS() {

        while((!wait_LS_queue.empty()) && freeSM>0) {
            auto submit_time = wait_LS_queue.front();
            wait_LS_queue.erase(wait_LS_queue.begin());
            auto wait_time = cur_timestamp - submit_time;
            acc_latency += wait_time;

            freeSM--;
            served_SM++;

            auto job = execute_LS_queue->submit_and_run_kernel();
            ls_occupy.push_back(std::make_pair(cur_timestamp, job->needed_exec_time));
        }
    }

    void busy_check_queues() {
        for(const auto& batch_queue : batch_queues) {
            auto forward_info = batch_queue->busy_check_queue();
            freeSM += forward_info.first;
            //TODO deal non-PT kernel sperately
            acc_throughput += forward_info.second * batch_queue->normthrough();
        }

        auto forward_info = execute_LS_queue->busy_check_queue();
        freeSM += forward_info.first;
        finished_LS += forward_info.second;
    }



    bool _allocate_freeSM() {
        for(auto & batch_queue : batch_queues) {
            if(!batch_queue->full()) {
                if(batch_queue->try_resume()) {
                    // do nothing
                } else {
                    batch_queue->submit_and_run_kernel();
                }
                freeSM--;

                if (batch_queue->onfly > 0) {
                    batch_queue->onfly -= 1;
                }
                return true;
            }
        }
        return false;
    }

    void allocate_freeSM() {
        while (freeSM > 0) {
            if (!_allocate_freeSM()) {
                break;
            }
        }
    }

    virtual void proactive_preemt() {
        return;
    }

    void busy_check_system() {
        busy_check_queues();
        launch_waiting_LS();
        allocate_freeSM();
        proactive_preemt();

        cur_timestamp += 1;

    }

    void stop() {
        float remain_throughput = .0;
        for(auto& batch_queue : batch_queues) {
            remain_throughput += batch_queue->stop() * batch_queue->normthrough();
        }
        acc_throughput += remain_throughput;
    }

    void schedule_exp(std::vector<int> arrvial_times, int job_len) {
        int processed_LS_jobs = 0;
        while(served_SM < job_len) {
            if (processed_LS_jobs < job_len && arrvial_times[processed_LS_jobs] == cur_timestamp) {
                submit_LS(); //
                processed_LS_jobs++;
            }
            if(processed_LS_jobs < job_len) {
                test_period = cur_timestamp;
            }
            busy_check_system();
        }
        stop();
    }

    float get_latency() {
        return acc_latency;
    }

    float get_throughput() {
        return acc_throughput; 
        //this should be finished kernels, but we need to div the time
    }

    float get_norm_throughput() {
        //we stop at the last job have been submitted to GPU
        //it makes the LS jobs occupy more GPU time when latency is long
        //we need to minus this period for fairness
        //but actually, the influence is minor
        int norm_period = 0;
        for(int i = 0; i < ls_occupy.size(); ++i) {
            int start_time = ls_occupy[i].first;
            int end_time = ls_occupy[i].first + ls_occupy[i].second;
            if (end_time > cur_timestamp) {
                norm_period += (cur_timestamp - start_time);
            } else {
                norm_period += ls_occupy[i].second;
            }
        }
        norm_period = norm_period / 16;
        return float(acc_throughput) / (test_period + norm_period);
    }

};

class FlushingSys : public ScheduleSystem {
public:
    FlushingSys(std::string kernel_name, std::string LS_name) 
    : ScheduleSystem(kernel_name, LS_name) {
        auto queue0 = std::make_shared<BatchQueue>(kernel_name, 0, 0, 16);
        for(int i = 0; i < 16; ++i) {
            queue0->submit_and_run_kernel();
        }
        batch_queues.push_back(queue0);
    }
};

class DeferringSys : public ScheduleSystem {
public:
    DeferringSys(std::string kernel_name, std::string LS_name) 
    : ScheduleSystem(kernel_name, LS_name) {
        auto queue0 = std::make_shared<BatchQueue>(kernel_name, 2, 2, 16);
        for(int i = 0; i < 16; ++i) {
            queue0->submit_and_run_kernel();
        }
        batch_queues.push_back(queue0);
    }
};

class CPSpatialSys : public ScheduleSystem {
public:
    CPSpatialSys(std::string kernel_name, std::string LS_name) 
    : ScheduleSystem(kernel_name, LS_name) {

        int r0_size = 4, r1_size = 4, r2_size = 8;

        auto queue0 = std::make_shared<BatchQueue>(kernel_name+"_nopt", 0, 0, r0_size, 2);
        auto queue1 = std::make_shared<BatchQueue>(kernel_name+"_ckpt_pt", 1, 1, r1_size, 1);
        auto queue2 = std::make_shared<BatchQueue>(kernel_name+"_pt", 2, 2, r2_size, 0);

        for(int i = 0; i < r0_size; ++i) {
            queue0->submit_and_run_kernel();
        }
        for(int i = 0; i < r1_size; ++i) {
            queue1->submit_and_run_kernel();
        }
        for(int i = 0; i < r2_size; ++i) {
            queue2->submit_and_run_kernel();
        }
        batch_queues.push_back(queue0);
        batch_queues.push_back(queue1);
        batch_queues.push_back(queue2);

    }

    bool _proactive_preemt() {
        for(auto& batch_queue : batch_queues) {
            int next_eq_id = batch_queue->eq_id+1;
            if(next_eq_id < 3) {
                if(batch_queue->atom_queue.size() + batch_queue->onfly + freeSM < batch_queue->low_threshold) {
                    if(!batch_queues[next_eq_id]->empty()) {
                        batch_queues[next_eq_id]->dequeue();
                        batch_queue->onfly += 1;
                        return true;
                    }

                    if(batch_queues[next_eq_id]->onfly > 0) {
                        batch_queues[next_eq_id]->onfly -= 1;
                        batch_queue->onfly +=1;
                    }
                }
            }

        }  
        return false;
    }

    virtual void proactive_preemt() override {
        while (_proactive_preemt()) {}
    }
};




char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}



int main(int argc, char * argv[]) {

#ifdef DEBUG
    for(int i = 0; i < 6; ++i) {
        std::cout << "CAREFULLY! IN DEBUG MODE!!"
    }
#endif

    const std::string routine_dir = "../routines/";
    for (const auto & entry : std::filesystem::directory_iterator(routine_dir)) {
        std::string path_name = entry.path();
        
        std::string end = path_name.substr(path_name.find_last_of("."), path_name.size());
        if(end != ".routine") {
            continue;
        }
        std::string index = path_name.substr(path_name.find_last_of("/")+1, path_name.size());
        routine_info_global[index] = new KernelInfo(
            index,
            std::vector<size_t>{64},
            std::vector<size_t>{64},
            {
                std::make_pair<size_t, std::shared_ptr<void*>>(sizeof(cl_mem), handler.create_buffer<float>(1024))
            },
            JobType::ROUTINE
        );
    }
    

    int test_num = 7;
    if(cmdOptionExists(argv, argv+argc, "-fullexp")) {
        std::vector<std::vector<int>> arrvial_times;
        arrvial_times.resize(test_num);
        for(int i = 0; i < test_num; ++i) {
            auto interval = 20 * pow(4, i);
            std::default_random_engine generator;
            std::poisson_distribution<int> distribution(interval);
            float t = 0;
            for(int j = 0; j < 1000; ++j) {
                t += float(std::abs(distribution(generator)));
                arrvial_times[i].push_back(static_cast<int>(t * TIMESTAMP_GRAIN));
            }
        }

        std::string test_case = "vecadd";
        for(int test_iter = 0; test_iter < test_num; ++test_iter) {
            
            std::cout << "*****************************************" << std::endl;

            std::cout << "Experiment: test group " << test_iter << std::endl;
            
            auto noptds = FlushingSys("vecadd_nopt", "vecadd_LS");
            noptds.schedule_exp(arrvial_times[test_iter], 16);
            
            std::cout << "NOPT-DS -> latency: " << (noptds.get_latency() / TIMESTAMP_GRAIN / 16)
                << ", norm_throughput: " << noptds.get_norm_throughput() << std::endl;
            
            auto ptds = FlushingSys("vecadd_pt", "vecadd_LS");
            ptds.schedule_exp(arrvial_times[test_iter], 16);
            
            std::cout << "PT-DS -> latency: " << (ptds.get_latency() / TIMESTAMP_GRAIN / 16)
                << ", norm_throughput: " << ptds.get_norm_throughput() << std::endl;

            auto cpspatial = CPSpatialSys("vecadd", "vecadd_LS");
            cpspatial.schedule_exp(arrvial_times[test_iter], 16);
            
            std::cout << "CPSpatial -> latency: " << (cpspatial.get_latency() / TIMESTAMP_GRAIN / 16)
                << ", norm_throughput: " << cpspatial.get_norm_throughput() << std::endl;

            auto ptdf = DeferringSys("vecadd_pt", "vecadd_LS");
            ptdf.schedule_exp(arrvial_times[test_iter], 16);
            
            std::cout << "PT+Defer -> latency: " << (ptdf.get_latency() / TIMESTAMP_GRAIN / 16)
                << ", norm_throughput: " << ptdf.get_norm_throughput() << std::endl;
            std::cout << "*****************************************" << std::endl;
            std::cout << " " << std::endl;

        }
    } else if (cmdOptionExists(argv, argv+argc, "-interval")) {
        char* interval_ctr = getCmdOption(argv, argv + argc, "-interval");
        int interval = atoi(interval_ctr);
        
        std::default_random_engine generator;
        std::poisson_distribution<int> distribution(interval);

        std::vector<int> arrvial_time;
        float t = 0;
        for(int j = 0; j < 1000; ++j) {
            // t += float(interval);
            t += float(std::abs(distribution(generator)));
            arrvial_time.push_back(static_cast<int>(t * TIMESTAMP_GRAIN));
        }

        std::string test_case = "vecadd";
            
        std::cout << "*****************************************" << std::endl;

        std::cout << "Experiment: interval = " << interval << std::endl;
        
        auto noptds = FlushingSys("vecadd_nopt", "vecadd_LS");
        noptds.schedule_exp(arrvial_time, 16);
        
        std::cout << "NOPT-DS -> latency: " << (noptds.get_latency() / TIMESTAMP_GRAIN / 16)
            << ", norm_throughput: " << noptds.get_norm_throughput() << std::endl;
        
        auto ptds = FlushingSys("vecadd_pt", "vecadd_LS");
        ptds.schedule_exp(arrvial_time, 16);
        
        std::cout << "PT-DS -> latency: " << (ptds.get_latency() / TIMESTAMP_GRAIN / 16)
            << ", norm_throughput: " << ptds.get_norm_throughput() << std::endl;

        auto cpspatial = CPSpatialSys("vecadd", "vecadd_LS");
        cpspatial.schedule_exp(arrvial_time, 16);
        
        std::cout << "CPSpatial -> latency: " << (cpspatial.get_latency() / TIMESTAMP_GRAIN / 16)
            << ", norm_throughput: " << cpspatial.get_norm_throughput() << std::endl;

        auto ptdf = DeferringSys("vecadd_pt", "vecadd_LS");
        ptdf.schedule_exp(arrvial_time, 16);
        
        std::cout << "PT+Defer -> latency: " << (ptdf.get_latency() / TIMESTAMP_GRAIN / 16)
            << ", norm_throughput: " << ptdf.get_norm_throughput() << std::endl;
        std::cout << "*****************************************" << std::endl;
        std::cout << " " << std::endl;


    }

    


#ifdef DEBUG
    for(int i = 0; i < 6; ++i) {
        std::cout << "CAREFULLY! IN DEBUG MODE!!"
    }
#else
    
#endif
}