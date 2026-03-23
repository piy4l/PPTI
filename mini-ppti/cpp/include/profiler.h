#ifndef PROFILER_H
#define PROFILER_H

#include <chrono>
#include <map>
#include <string>
#include <vector>

struct ProfileRecord {
    std::string op_name;
    double time_ms;
    std::size_t input_size;
    std::string metadata;
};

class Profiler {
public:
    void add_record(const ProfileRecord& record);
    void print_summary() const;
    void print_csv() const;

private:
    std::vector<ProfileRecord> records_;
};

class ScopedTimer {
public:
    ScopedTimer(Profiler& profiler,
                const std::string& op_name,
                std::size_t input_size,
                const std::string& metadata = "");

    ~ScopedTimer();

private:
    Profiler& profiler_;
    std::string op_name_;
    std::size_t input_size_;
    std::string metadata_;
    std::chrono::high_resolution_clock::time_point start_;
};

#endif