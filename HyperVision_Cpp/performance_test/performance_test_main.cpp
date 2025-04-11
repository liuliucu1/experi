#include "performance_tester.hpp"
#include <iostream>
#include <string>
#include <cstdlib>

using namespace Hypervision;

void print_usage() {
    std::cout << "Usage: performance_test [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --num-experiments N    Number of experiments to run (default: 10)" << std::endl;
    std::cout << "  --num-packets N        Number of packets to process (default: 10000)" << std::endl;
    std::cout << "  --data PATH            Path to data file (default: /home/sduu2/userspace-20T-1/yyr/data/dns_lrscan.data)" << std::endl;
    std::cout << "  --label PATH           Path to label file (default: /home/sduu2/userspace-20T-1/yyr/data/dns_lrscan.label)" << std::endl;
    std::cout << "  --max-packets N        Maximum number of packets to process (default: 1000000)" << std::endl;
    std::cout << "  --output PATH          Path to output CSV file (default: performance_results.csv)" << std::endl;
    std::cout << "  --test TYPE            Test type: all, baseline, dynamic, priority, realtime (default: all)" << std::endl;
    std::cout << "  --help                 Display this help message" << std::endl;
}

int main(int argc, char** argv) {
    // Default parameters
    size_t num_experiments = 10;
    size_t num_packets = 10000;
    std::string data_path = "/home/sduu2/userspace-20T-1/yyr/data/dns_lrscan.data";
    std::string label_path = "/home/sduu2/userspace-20T-1/yyr/data/dns_lrscan.label";
    size_t max_packets = 1000000;
    std::string output_path = "performance_results.csv";
    std::string test_type = "all";
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage();
            return 0;
        } else if (arg == "--num-experiments" && i + 1 < argc) {
            num_experiments = std::atoi(argv[++i]);
        } else if (arg == "--num-packets" && i + 1 < argc) {
            num_packets = std::atoi(argv[++i]);
        } else if (arg == "--data" && i + 1 < argc) {
            data_path = argv[++i];
        } else if (arg == "--label" && i + 1 < argc) {
            label_path = argv[++i];
        } else if (arg == "--max-packets" && i + 1 < argc) {
            max_packets = std::atoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--test" && i + 1 < argc) {
            test_type = argv[++i];
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage();
            return 1;
        }
    }
    
    // Create performance tester
    PerformanceTester tester(num_experiments, num_packets, data_path, label_path, output_path, max_packets);
    
    // Run tests based on test type
    if (test_type == "all") {
        std::cout << "Running all performance tests..." << std::endl;
        tester.run_all_tests();
    } else if (test_type == "baseline") {
        std::cout << "Running baseline clustering test..." << std::endl;
        auto metrics = tester.test_baseline_performance();
        std::cout << "Processing time: " << metrics.processing_time << " seconds" << std::endl;
        std::cout << "Throughput: " << metrics.throughput << " packets/second" << std::endl;
    } else if (test_type == "dynamic") {
        std::cout << "Running dynamic clustering test..." << std::endl;
        auto metrics = tester.test_dynamic_clustering_performance();
        std::cout << "Processing time: " << metrics.processing_time << " seconds" << std::endl;
        std::cout << "Throughput: " << metrics.throughput << " packets/second" << std::endl;
        std::cout << "Memory usage: " << metrics.memory_usage << " MB" << std::endl;
    } else if (test_type == "priority") {
        std::cout << "Running priority queue test..." << std::endl;
        auto metrics = tester.test_priority_queue_performance();
        std::cout << "Processing time: " << metrics.processing_time << " seconds" << std::endl;
        std::cout << "Throughput: " << metrics.throughput << " packets/second" << std::endl;
        std::cout << "Memory usage: " << metrics.memory_usage << " MB" << std::endl;
    } else if (test_type == "realtime") {
        std::cout << "Running realtime processing test..." << std::endl;
        auto metrics = tester.test_realtime_performance();
        std::cout << "Processing time: " << metrics.processing_time << " seconds" << std::endl;
        std::cout << "Throughput: " << metrics.throughput << " packets/second" << std::endl;
        std::cout << "Memory usage: " << metrics.memory_usage << " MB" << std::endl;
    } else {
        std::cerr << "Unknown test type: " << test_type << std::endl;
        print_usage();
        return 1;
    }
    
    return 0;
}
