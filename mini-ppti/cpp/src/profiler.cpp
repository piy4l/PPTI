#include "profiler.h"

#include <iomanip>
#include <iostream>

void Profiler::add_record(const ProfileRecord& record) {
    records_.push_back(record);
}

const std::vector<ProfileRecord>& Profiler::get_records() const {
    return records_;
}

void Profiler::print_summary() const {
    std::cout << "\n=== Profiling Summary ===\n";
    std::cout << std::left
              << std::setw(20) << "Operation"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Input Size"
              << "Metadata\n";

    for (const auto& record : records_) {
        std::cout << std::left
                  << std::setw(20) << record.op_name
                  << std::setw(15) << std::fixed << std::setprecision(6) << record.time_ms
                  << std::setw(15) << record.input_size
                  << record.metadata << "\n";
    }
}

void Profiler::print_csv() const {
    std::cout << "op_name,time_ms,input_size,metadata\n";
    for (const auto& record : records_) {
        std::cout << record.op_name << ","
                  << std::fixed << std::setprecision(6) << record.time_ms << ","
                  << record.input_size << ","
                  << "\"" << record.metadata << "\"\n";
    }
}

ScopedTimer::ScopedTimer(Profiler& profiler,
                         const std::string& op_name,
                         std::size_t input_size,
                         const std::string& metadata)
    : profiler_(profiler),
      op_name_(op_name),
      input_size_(input_size),
      metadata_(metadata),
      start_(std::chrono::high_resolution_clock::now()) {}

ScopedTimer::~ScopedTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start_;

    profiler_.add_record({
        op_name_,
        elapsed.count(),
        input_size_,
        metadata_
    });
}