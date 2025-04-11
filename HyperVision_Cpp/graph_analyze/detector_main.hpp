#pragma once
#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
#include <direct.h> // For _mkdir on Windows
#endif
#include "../packet_parse/pcap_parser.hpp"
#include "../flow_construct/explicit_constructor.hpp"
#include "../flow_construct/flow_define.hpp"
#include "edge_constructor.hpp"
#include "graph_define.hpp"
#include <filesystem> // Using filesystem for path manipulation is generally better if C++17 is available
#include <cmath>
#include <numeric> // For std::accumulate
#include <algorithm>
#include <chrono>
#include <queue>
#include <iomanip> // For setprecision

// Using filesystem namespace if available, otherwise stick to string manipulation
#if __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
namespace fs = std::filesystem;
#endif

// Assume LOGF, FATAL_ERROR, __START_FTIMMER__, etc. are defined elsewhere
#ifndef LOGF
#define LOGF(...) printf(__VA_ARGS__); printf("\n") // Basic printf logging if not defined
#endif
#ifndef FATAL_ERROR
#define FATAL_ERROR(msg) { printf("FATAL ERROR: %s\n", msg.c_str()); exit(1); }
#endif


namespace Hypervision
{

class hypervision_detector {
private:

    json jin_main;
    string file_path = "";

    shared_ptr<vector<shared_ptr<basic_packet> > > p_parse_result;

    shared_ptr<binary_label_t> p_label; // Overall label if loaded initially
    shared_ptr<vector<double_t> > p_loss; // Holds loss for the *last* processed window of the *last* dataset

    shared_ptr<vector<shared_ptr<basic_flow> > > p_flow;

    shared_ptr<vector<shared_ptr<short_edge> > > p_short_edges;
    shared_ptr<vector<shared_ptr<long_edge> > > p_long_edges;

    // These vectors will store results PER DATASET before being cleared
    vector<pair<shared_ptr<binary_label_t>, shared_ptr<vector<double_t>>>> window_results;
    vector<double> window_processing_times;
    unordered_map<string, vector<double>> module_durations;

    vector<string> module_names = {
        "data_preparation",
        "flow_construct",
        "edge_construct",
        "graph_parse",
        "incremental_learning",
        "graph_detect",
        "final_score_calc"
    };

    bool save_result_enable = false;
    string save_result_path = "../temp/default.json"; // Treat this as base path/dir

    // Sliding window parameters
    size_t window_size = 900000;

    // Incremental learning parameters
    bool incremental_learning_enabled = true;
    string model_save_path = "../models/";
    bool load_previous_model = false;
    string model_load_path = ""; // Path to load the *initial* model from
    double learning_rate = 0.01;
    double model_decay_factor = 0.9;

    // Dynamic update parameters
    bool dynamic_updates_enabled = false;
    size_t min_packets_for_update = 50;
    double packet_priority_threshold = 0.7;

    // Multi-dataset parameters
    bool multi_dataset_mode = false;
    vector<string> dataset_data_paths;
    vector<string> dataset_label_paths;
    size_t current_dataset_index = 0;

    // Real-time processing parameters
    bool real_time_mode = false;
    double max_processing_delay = 0.1;
    shared_ptr<traffic_graph> current_graph_model = nullptr; // Used for dynamic updates

    // Helper for directory creation
    bool create_directory_if_not_exists(const string& path) const {
        #if __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
            try {
                if (!fs::exists(path)) {
                    if (fs::create_directories(path)) {
                         LOGF("Created directory: %s", path.c_str());
                         return true;
                    } else {
                         LOGF("Failed to create directory: %s", path.c_str());
                         return false;
                    }
                } else if (!fs::is_directory(path)) {
                    LOGF("Path exists but is not a directory: %s", path.c_str());
                    return false;
                }
                 return true; // Directory already exists
            } catch (const std::exception& e) {
                 LOGF("Filesystem error checking/creating directory %s: %s", path.c_str(), e.what());
                 return false;
            }
        #else
            struct stat info;
            if (stat(path.c_str(), &info) != 0) {
                #if defined(_WIN32)
                    if (_mkdir(path.c_str()) == 0) {
                        LOGF("Created directory: %s", path.c_str());
                        return true;
                    } else {
                        LOGF("Failed to create directory (mkdir error): %s", path.c_str());
                        return false;
                    }
                #else
                    if (mkdir(path.c_str(), 0755) == 0) { // Linux/macOS
                        LOGF("Created directory: %s", path.c_str());
                        return true;
                    } else {
                        LOGF("Failed to create directory (mkdir error): %s", path.c_str());
                        return false;
                    }
                #endif
            } else if (!S_ISDIR(info.st_mode)) {
                 LOGF("Path exists but is not a directory: %s", path.c_str());
                 return false;
            }
             return true;
        #endif
    }


public:
    void start(void) {
        __START_FTIMMER__

        // --- Determine Base Save Directory ---
        string base_save_dir;
        size_t last_slash = save_result_path.find_last_of("/\\");
        if (last_slash != string::npos) {
             size_t last_dot = save_result_path.find_last_of('.');
             if (last_dot != string::npos && last_dot > last_slash) {
                 base_save_dir = save_result_path.substr(0, last_slash);
             } else {
                 base_save_dir = save_result_path;
             }
        } else {
             base_save_dir = ".";
        }
        if (!base_save_dir.empty() && base_save_dir != "." && base_save_dir.back() != '/' && base_save_dir.back() != '\\') {
            base_save_dir += '/';
        }

        // --- Model Pointer Initialization ---
        // This pointer will hold the model state across datasets/runs
        shared_ptr<traffic_graph> current_run_model = nullptr;

        // Load initial model if configured (applies to both single and multi-dataset start)
        if (incremental_learning_enabled && load_previous_model && !model_load_path.empty()) {
            LOGF("Attempting to load initial base model from: %s", model_load_path.c_str());
            current_run_model = load_model(model_load_path); // Load into the persistent pointer
            if (current_run_model) {
                LOGF("Successfully loaded initial base model.");
                 // If dynamic updates are also enabled, use this loaded model initially
                if (dynamic_updates_enabled) {
                    current_graph_model = current_run_model;
                    current_graph_model->set_dynamic_updates(true);
                     LOGF("Setting loaded model for dynamic updates.");
                }
            } else {
                LOGF("Failed to load initial base model from %s, will train from scratch.", model_load_path.c_str());
            }
        }


        // --- Processing Logic ---
        if (multi_dataset_mode && !dataset_data_paths.empty() && !dataset_label_paths.empty()) {
            // --- Multi-Dataset Mode ---
            LOGF("Running in multi-dataset mode with %zu datasets. Base save directory: %s", dataset_data_paths.size(), base_save_dir.c_str());
            LOGF("Chained incremental learning is %s.", incremental_learning_enabled ? "ENABLED" : "DISABLED");

            for (size_t dataset_idx = 0; dataset_idx < dataset_data_paths.size(); dataset_idx++) {
                current_dataset_index = dataset_idx; // Update index for logging/saving
                string data_path = dataset_data_paths[dataset_idx];
                string label_path = dataset_label_paths[dataset_idx];

                LOGF("--- Starting Dataset %zu/%zu: %s ---", dataset_idx + 1, dataset_data_paths.size(), data_path.c_str());
                if (current_run_model && incremental_learning_enabled) {
                     LOGF("Using model from previous dataset/initial load as starting point.");
                } else if (incremental_learning_enabled) {
                     LOGF("No previous model available, starting fresh for this dataset.");
                }

                // Load the current dataset's data
                const auto p_dataset_constructor = make_shared<basic_dataset>();
                p_dataset_constructor->configure_via_json(jin_main["dataset_construct"]);
                p_dataset_constructor->set_data_path(data_path);
                p_dataset_constructor->set_label_path(label_path);
                p_dataset_constructor->import_dataset();
                p_label = p_dataset_constructor->get_label();
                p_parse_result = p_dataset_constructor->get_raw_pkt();

                // *** Process this dataset, passing the current model state ***
                // `process_dataset` will use `current_run_model` as its starting point
                // and update `current_run_model` with the result after processing this dataset.
                process_dataset(current_run_model);

                // Save results (labels, scores, timings) for THIS dataset
                if (save_result_enable) {
                    string dataset_save_dir = base_save_dir + "dataset_" + to_string(dataset_idx);
                    LOGF("Saving results for dataset %zu to: %s", dataset_idx, dataset_save_dir.c_str());
                    do_save2(dataset_save_dir); // Pass the specific directory
                }

                // Clear per-dataset results and intermediate data structures
                // *** Crucially, current_run_model is NOT cleared here ***
                 window_results.clear();
                 window_processing_times.clear();
                 for (auto& pair : module_durations) {
                     pair.second.clear();
                 }
                 p_flow.reset();
                 p_short_edges.reset();
                 p_long_edges.reset();
                 p_loss.reset(); // Only holds last window result anyway
                 p_label.reset(); // Release labels for previous dataset
                 p_parse_result.reset(); // Release packets for previous dataset
                 LOGF("Cleared window results and intermediate data for dataset %zu.", dataset_idx);
                 LOGF("--- Finished Dataset %zu/%zu ---", dataset_idx + 1, dataset_data_paths.size());

            } // End dataset loop

            LOGF("Completed processing all %zu datasets.", dataset_data_paths.size());
            if (current_run_model && incremental_learning_enabled) {
                 // Optionally save the final chained model after all datasets
                 string final_model_path = model_save_path + "/final_chained_model.bin";
                 LOGF("Saving final chained model after all datasets to: %s", final_model_path.c_str());
                 save_model(current_run_model, final_model_path);
            }

        } else {
            // --- Single Dataset Mode ---
            LOGF("Running in single dataset mode.");
            if (jin_main.count("packet_parse") &&
                jin_main["packet_parse"].count("target_file_path")) {

                LOGF("Parse packet from file: %s", jin_main["packet_parse"]["target_file_path"].get<string>().c_str());
                file_path = jin_main["packet_parse"]["target_file_path"];
                const auto p_packet_parser = make_shared<pcap_parser>(file_path);
                p_packet_parser->parse_raw_packet();
                p_packet_parser->parse_basic_packet_fast();
                p_parse_result = p_packet_parser->get_basic_packet_rep();

                const auto p_dataset_constructor = make_shared<basic_dataset>(p_parse_result);
                p_dataset_constructor->configure_via_json(jin_main["dataset_construct"]);
                p_dataset_constructor->do_dataset_construct();
                p_label = p_dataset_constructor->get_label();

            } else if (jin_main["dataset_construct"].count("data_path") &&
                       jin_main["dataset_construct"].count("label_path")){
                LOGF("Load dataset from data_path: %s, label_path: %s",
                     jin_main["dataset_construct"]["data_path"].get<string>().c_str(),
                     jin_main["dataset_construct"]["label_path"].get<string>().c_str());
                p_parse_result = make_shared<vector<shared_ptr<basic_packet>>>(); // Initialize if loading directly
                const auto p_dataset_constructor = make_shared<basic_dataset>(p_parse_result);
                p_dataset_constructor->configure_via_json(jin_main["dataset_construct"]);
                p_dataset_constructor->import_dataset();
                p_label = p_dataset_constructor->get_label();
                p_parse_result = p_dataset_constructor->get_raw_pkt();

            } else {
                LOGF("Dataset source not configured correctly for single dataset mode.");
                return;
            }

             // *** Process the single dataset ***
             // Pass the initially loaded (or null) model. It will be updated if IL is enabled.
             LOGF("Processing single dataset...");
             process_dataset(current_run_model);
             LOGF("Finished processing single dataset.");


            // Save results ONCE for the single dataset run
            if (save_result_enable) {
                LOGF("Saving results for single dataset run to directory: %s", base_save_dir.c_str());
                do_save2(base_save_dir); // Save to the base directory
            }

             // Optionally save the final model for the single run
             if (current_run_model && incremental_learning_enabled) {
                 string final_model_path = model_save_path + "/final_single_run_model.bin";
                 LOGF("Saving final model after single dataset run to: %s", final_model_path.c_str());
                 save_model(current_run_model, final_model_path);
             }
        }

        __STOP_FTIMMER__
        __PRINTF_EXE_TIME__
    }


    // **** MODIFIED process_dataset Signature and Logic ****
    // Accepts the current cumulative model state and updates it.
    void process_dataset(shared_ptr<traffic_graph>& model_in_out) {
        // Initialize module timings for this dataset run
        for (const auto& module : module_names) {
            module_durations[module] = vector<double>();
        }
        window_results.clear();
        window_processing_times.clear();

        if (!p_parse_result) {
             FATAL_ERROR("Packet data (p_parse_result) is null during process_dataset.");
             return;
        }
        size_t total_data_size = p_parse_result->size();
        LOGF("Processing dataset (Index: %zu) with size: %zu", current_dataset_index, total_data_size);

        if (total_data_size == 0) {
            LOGF("Warning: Dataset is empty. Skipping processing.");
            // Ensure model_in_out remains unchanged if dataset is empty
            return;
        }

        size_t effective_window_size = window_size; // Use a local copy for calculations
        size_t num_windows;
        if (effective_window_size == 0) {
             LOGF("Warning: Window size is zero. Processing entire dataset as one window.");
             num_windows = 1;
             effective_window_size = total_data_size;
        } else if (total_data_size < effective_window_size) {
            LOGF("Info: Dataset size (%zu) smaller than window size (%zu). Processing as one window.", total_data_size, effective_window_size);
            num_windows = 1;
            // Use full data size for the single window's end index calculation
            effective_window_size = total_data_size;
        } else {
            num_windows = (total_data_size + effective_window_size - 1) / effective_window_size; // Ceiling division for potentially partial last window
             // Note: Original code used floor division `total_data_size / window_size`.
             // Using ceiling ensures the last packets are processed if they form a partial window.
             // Adjust if only *full* windows were intended. Assuming we want to process all data.
             LOGF("Using sliding window approach. Window size: %zu, Num windows (incl. partial): %zu",
                  effective_window_size, num_windows);
        }


        // --- Model Chaining Logic ---
        // `previous_model_for_window` tracks the model state *between windows within this dataset*.
        // It starts with the state passed *into* this function (`model_in_out`).
        shared_ptr<traffic_graph> previous_model_for_window = model_in_out;
        if (incremental_learning_enabled) {
            if (previous_model_for_window) {
                 LOGF("Starting dataset %zu processing with model from previous dataset/load.", current_dataset_index);
            } else {
                 LOGF("Starting dataset %zu processing fresh (no previous model).", current_dataset_index);
            }
        }


        // Process each window
        for (size_t window_idx = 0; window_idx < num_windows; ++window_idx) {
            auto window_start_time = std::chrono::high_resolution_clock::now();

            // **** Cumulative Window Logic ****
            // Windows grow: Window 0 = [0, W], Window 1 = [0, 2W], Window 2 = [0, 3W]...
            // If non-cumulative windows are needed, adjust start_idx = window_idx * effective_window_size
            size_t start_idx = 0;
            size_t train_end_idx = std::min((window_idx + 1) * effective_window_size, total_data_size);
             size_t current_window_pkt_count = train_end_idx - start_idx;

            LOGF("Window %zu (Dataset %zu): Processing packets [%zu-%zu] (Size: %zu)",
                 window_idx, current_dataset_index, start_idx, train_end_idx, current_window_pkt_count);

            if (current_window_pkt_count == 0) {
                LOGF("Skipping empty window %zu.", window_idx);
                continue; // Skip if somehow window size is zero
            }

            // --- Data Preparation ---
            // (Same as before, prepares data for the current cumulative window)
            auto data_prep_start = std::chrono::high_resolution_clock::now();
            auto train_data = make_shared<vector<shared_ptr<basic_packet>>>(
                p_parse_result->begin() + start_idx,
                p_parse_result->begin() + train_end_idx
            );
            auto window_label = make_shared<binary_label_t>(); // Labels corresponding to train_data
            window_label->resize(current_window_pkt_count);
            if (p_label && p_label->size() >= train_end_idx) {
                std::copy(p_label->begin() + start_idx, p_label->begin() + train_end_idx, window_label->begin());
            } else if (p_label) { // Handle cases where labels might be short
                 size_t available_labels = p_label->size() > start_idx ? p_label->size() - start_idx : 0;
                 size_t copy_count = std::min(available_labels, window_label->size());
                 std::copy(p_label->begin() + start_idx, p_label->begin() + start_idx + copy_count, window_label->begin());
                 if (copy_count < window_label->size()) {
                      LOGF("Warning: Label data shorter than packet data for window %zu. Copied %zu labels, expected %zu.", window_idx, copy_count, window_label->size());
                 }
            }
            const auto p_flow_constructor = make_shared<explicit_flow_constructor>(train_data);
            p_flow_constructor->config_via_json(jin_main["flow_construct"]);
            auto data_prep_end = std::chrono::high_resolution_clock::now();
            module_durations["data_preparation"].push_back(std::chrono::duration<double>(data_prep_end - data_prep_start).count());
            // LOGF("Data prep time W%zu: %.3fs", window_idx, module_durations["data_preparation"].back());


            // --- Flow Construct ---
            auto flow_start = std::chrono::high_resolution_clock::now();
            p_flow_constructor->construct_flow();
            p_flow = p_flow_constructor->get_constructed_raw_flow();
            auto flow_end = std::chrono::high_resolution_clock::now();
            module_durations["flow_construct"].push_back(std::chrono::duration<double>(flow_end - flow_start).count());
            // LOGF("Flow construct time W%zu: %.3fs", window_idx, module_durations["flow_construct"].back());


            // --- Edge Construct ---
            const auto p_edge_constructor = make_shared<edge_constructor>(p_flow);
            p_edge_constructor->config_via_json(jin_main["edge_construct"]);
            auto edge_start = std::chrono::high_resolution_clock::now();
            p_edge_constructor->do_construct();
            tie(p_short_edges, p_long_edges) = p_edge_constructor->get_edge();
            auto edge_end = std::chrono::high_resolution_clock::now();
            module_durations["edge_construct"].push_back(std::chrono::duration<double>(edge_end - edge_start).count());
            // LOGF("Edge construct time W%zu: %.3fs", window_idx, module_durations["edge_construct"].back());


            // --- Graph Parse ---
            // p_graph represents the graph built *purely from the current window's data*
            const auto p_graph = make_shared<traffic_graph>(p_short_edges, p_long_edges);
            p_graph->config_via_json(jin_main["graph_analyze"]);
            if (dynamic_updates_enabled) { // Dynamic updates apply to the latest model state
                p_graph->set_dynamic_updates(true);
                // Note: current_graph_model in the class scope should probably be updated
                // after incremental learning if dynamic updates are needed on the combined model.
                // Let's update it after the IL step.
            }
            auto graph_start = std::chrono::high_resolution_clock::now();
            p_graph->parse_edge(); // Parses edges from the current window's data
            auto graph_end = std::chrono::high_resolution_clock::now();
            module_durations["graph_parse"].push_back(std::chrono::duration<double>(graph_end - graph_start).count());
            // LOGF("Graph parse time W%zu: %.3fs", window_idx, module_durations["graph_parse"].back());


            // --- Incremental Learning Step ---
            auto incr_start = std::chrono::high_resolution_clock::now();
            if (incremental_learning_enabled) {
                if (previous_model_for_window) {
                    // Apply incremental learning: update p_graph (current window's graph)
                    // using the model state from the previous window/dataset (`previous_model_for_window`)
                    LOGF("Applying incremental learning for window %zu using previous state.", window_idx);
                    apply_incremental_learning(p_graph, previous_model_for_window);
                } else {
                    LOGF("Window %zu: No previous model state, using only current window data.", window_idx);
                    // No update needed, p_graph is the base model for now.
                }
                // *** Update the model for the NEXT window ***
                // The result of processing this window (potentially updated p_graph)
                // becomes the input model for the next window.
                previous_model_for_window = p_graph;

                // If dynamic updates are enabled, point the class member to the latest combined model state
                if (dynamic_updates_enabled) {
                    current_graph_model = previous_model_for_window;
                }
            } else {
                // If IL is disabled, we don't chain models between windows.
                // For consistency, we could still set previous_model_for_window = p_graph,
                // but it won't be used for updating in the next iteration.
                 previous_model_for_window = p_graph; // Keep track of the latest window's model
                 if (dynamic_updates_enabled) {
                     current_graph_model = previous_model_for_window; // Point to latest window model
                 }
            }
            auto incr_end = std::chrono::high_resolution_clock::now();
            module_durations["incremental_learning"].push_back(std::chrono::duration<double>(incr_end - incr_start).count());
             // LOGF("Incremental learning step time W%zu: %.3fs", window_idx, module_durations["incremental_learning"].back());


            // --- Graph Detect ---
            // Detection runs on the potentially updated graph (`previous_model_for_window` now holds the correct state)
            LOGF("Detecting for window %zu.", window_idx);
            auto detect_start = std::chrono::high_resolution_clock::now();
            if (previous_model_for_window) { // Should always be true after the logic above unless window was empty
                previous_model_for_window->graph_detect();
            } else {
                 LOGF("Error: Model is null before detection step in window %zu.", window_idx);
                 // Handle error case, maybe skip detection? Or use p_graph if IL was disabled?
                 // Sticking with previous_model_for_window as it should hold the correct state.
            }
            auto detect_end = std::chrono::high_resolution_clock::now();
            module_durations["graph_detect"].push_back(std::chrono::duration<double>(detect_end - detect_start).count());
            // LOGF("Detection time W%zu: %.3fs", window_idx, module_durations["graph_detect"].back());


            // --- Final Score Calculation ---
             LOGF("Calculating final score for window %zu.", window_idx);
             auto score_start = std::chrono::high_resolution_clock::now();
             shared_ptr<vector<double_t>> current_window_loss = nullptr;
             if (previous_model_for_window) {
                 // Get scores relative to the packets *in this window* using the final model state for this window
                 current_window_loss = previous_model_for_window->get_final_pkt_score(window_label);
             } else {
                  LOGF("Error: Model is null before score calculation in window %zu.", window_idx);
                  current_window_loss = make_shared<vector<double_t>>(current_window_pkt_count, 0.0); // Placeholder scores
             }
             // Store results for this window (labels and scores specific to this window's data)
             window_results.push_back({window_label, current_window_loss});
             // Update the class p_loss member to hold the latest window's scores (for potential legacy use)
             p_loss = current_window_loss;

             auto score_end = std::chrono::high_resolution_clock::now();
             module_durations["final_score_calc"].push_back(std::chrono::duration<double>(score_end - score_start).count());
             // LOGF("Score calculation time W%zu: %.3fs", window_idx, module_durations["final_score_calc"].back());


            // Calculate and store processing time for this window
            auto window_end_time = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double>(window_end_time - window_start_time).count();
            window_processing_times.push_back(duration);
            LOGF("Window %zu (Dataset %zu) total processing time: %.3f seconds", window_idx, current_dataset_index, duration);


            // --- Save Per-Window Model (if enabled) ---
            if (incremental_learning_enabled && previous_model_for_window) {
                string model_filename = "dataset_" + to_string(current_dataset_index) + "_window_" + to_string(window_idx) + "_model.bin";
                 if (create_directory_if_not_exists(model_save_path)) {
                    string model_path = model_save_path + "/" + model_filename;
                    // Save the state *after* processing this window
                    save_model(previous_model_for_window, model_path);
                 } else {
                     LOGF("Error: Could not create or access model save directory: %s. Skipping model save.", model_save_path.c_str());
                 }
            }

            LOGF("Completed processing for window %zu (Dataset %zu).", window_idx, current_dataset_index);

        } // End of window loop for the current dataset


        // --- Update Output Model ---
        // After processing all windows for this dataset, update the model state
        // that was passed by reference, so the next dataset (if any) gets the updated model.
        model_in_out = previous_model_for_window;
        LOGF("Finished processing all windows for dataset %zu. Updated output model reference.", current_dataset_index);

    } // End process_dataset


    // Method to process real-time packet data (unchanged)
    void process_packet(shared_ptr<basic_packet> packet) {
        if (!real_time_mode || !dynamic_updates_enabled || !current_graph_model) {
             if (!p_parse_result) p_parse_result = make_shared<vector<shared_ptr<basic_packet>>>();
            p_parse_result->push_back(packet);
            return;
        }
        // Use current_graph_model which should point to the latest model state
        current_graph_model->add_packet_to_queue(packet);
    }

    // Method to process a batch of packets in real-time mode (unchanged)
    void process_packet_batch(const vector<shared_ptr<basic_packet>>& packets) {
        if (!real_time_mode || !dynamic_updates_enabled || !current_graph_model) {
            if (!p_parse_result) p_parse_result = make_shared<vector<shared_ptr<basic_packet>>>();
            p_parse_result->insert(p_parse_result->end(), packets.begin(), packets.end());
            return;
        }
        if (!current_graph_model) {
             LOGF("Warning: Real-time batch processing requested but current_graph_model is null.");
             return;
        }
        for (const auto& packet : packets) {
            current_graph_model->add_packet_to_queue(packet);
        }
    }

    // config_via_json remains the same
    void config_via_json(const json & jin) {
         try {
             if (!jin.count("dataset_construct") || !jin.count("flow_construct") ||
                 !jin.count("edge_construct") || !jin.count("graph_analyze") || !jin.count("result_save")) {
                 throw logic_error("Incomplete json configuration. Missing one or more main sections.");
             }
             jin_main = jin;

             const auto j_save = jin["result_save"];
             save_result_enable = j_save.value("save_result_enable", false);
             save_result_path = j_save.value("save_result_path", "../temp/default.json");

             if (jin.count("window_params")) {
                 const auto j_window = jin["window_params"];
                 window_size = j_window.value("window_size", 900000);
                 if (window_size == 0) LOGF("Warning: Configured window_size is 0.");
             }

             if (jin.count("incremental_learning")) {
                 const auto j_incr = jin["incremental_learning"];
                 incremental_learning_enabled = j_incr.value("enabled", true);
                 model_save_path = j_incr.value("model_save_path", "../models/");
                 load_previous_model = j_incr.value("load_previous_model", true);
                 model_load_path = j_incr.value("model_load_path", "../models/");
                 learning_rate = j_incr.value("learning_rate", 0.01);
                 model_decay_factor = j_incr.value("model_decay_factor", 0.9);
             }

             if (jin.count("dynamic_updates")) {
                 const auto j_dyn = jin["dynamic_updates"];
                 dynamic_updates_enabled = j_dyn.value("enabled", false);
                 min_packets_for_update = j_dyn.value("min_packets_for_update", 50);
                 packet_priority_threshold = j_dyn.value("packet_priority_threshold", 0.7);
             }

             if (jin.count("real_time_mode")) {
                 const auto j_rt = jin["real_time_mode"];
                 real_time_mode = j_rt.value("enabled", false);
                 max_processing_delay = j_rt.value("max_processing_delay", 0.1);
             }

             if (jin.count("multi_dataset")) {
                 const auto j_multi = jin["multi_dataset"];
                 multi_dataset_mode = j_multi.value("enabled", false);
                 if (j_multi.count("data_paths") && j_multi["data_paths"].is_array()) {
                     dataset_data_paths = j_multi["data_paths"].get<vector<string>>();
                 }
                 if (j_multi.count("label_paths") && j_multi["label_paths"].is_array()) {
                     dataset_label_paths = j_multi["label_paths"].get<vector<string>>();
                 }
                 if (multi_dataset_mode && (dataset_data_paths.empty() || dataset_data_paths.size() != dataset_label_paths.size())) {
                      LOGF("Warning: Multi-dataset mode enabled but data_paths/label_paths are empty or mismatched. Disabling multi-dataset mode.");
                      multi_dataset_mode = false;
                 }
             }

         } catch (const json::exception& e) {
             FATAL_ERROR("JSON configuration error: " + string(e.what()));
         } catch (const exception & e) {
             FATAL_ERROR("Configuration error: " + string(e.what()));
         }
    }


    // do_save (legacy) remains the same
    void do_save(const string & save_path) {
        // ... (implementation unchanged) ...
        __START_FTIMMER__
        LOGF("Executing legacy do_save method to %s", save_path.c_str());
         if (!p_label || !p_loss) {
              LOGF("Error in do_save: p_label or p_loss is null. Cannot save.");
              __STOP_FTIMMER__; __PRINTF_EXE_TIME__;
              return;
         }
         // Use last window's results stored in p_loss and corresponding p_label slice?
         // This method seems less relevant with windowing. p_loss only has last window's scores.
         // Let's assume it saves whatever is currently in p_label and p_loss, acknowledging the mismatch potential.
         size_t common_size = 0;
         if (p_label && p_loss) {
             common_size = std::min(p_label->size(), p_loss->size());
             if (p_label->size() != p_loss->size()) {
                  LOGF("Warning in do_save: Label size (%zu) != Loss size (%zu). Saving up to common size %zu.", p_label->size(), p_loss->size(), common_size);
             }
         } else {
              LOGF("Error in do_save: p_label or p_loss is null.");
              __STOP_FTIMMER__; __PRINTF_EXE_TIME__;
              return;
         }


        ofstream _f(save_path);
        if (_f.is_open()) {
            try {
                _f << fixed << setprecision(6);
                for (size_t i = 0; i < common_size; ++i) {
                    _f << p_label->at(i) << ' '<< p_loss->at(i) << '\n';
                    if (i > 0 && i % 10000 == 0) _f << flush;
                }
                 _f << flush;
            } catch(const std::out_of_range& oor) {
                 LOGF("Exception (out_of_range) during do_save. Index likely exceeded bounds (%zu).", common_size);
            } catch(const exception & e) {
                LOGF("Exception during do_save: %s", e.what());
            }
            _f.close();
        } else {
            LOGF("File Error opening %s for do_save.", save_path.c_str());
        }

        __STOP_FTIMMER__
        __PRINTF_EXE_TIME__
    }


    // do_save2 (saving window results) remains the same
    void do_save2(const string& save_dir) const {
        // ... (implementation unchanged, saves window_results for the current dataset) ...
        LOGF("Executing do_save2 for directory: %s", save_dir.c_str());
        if (!create_directory_if_not_exists(save_dir)) {
             LOGF("Error: Failed to create or access save directory: %s. Aborting save.", save_dir.c_str());
             // Avoid FATAL_ERROR here, just log and return
             return;
        }

        if (window_results.empty() && window_processing_times.empty() && module_durations.empty()) {
            LOGF("No results or timing data found to save in %s.", save_dir.c_str());
            return;
        }

        // Save Timing Info
        const string timing_filename = save_dir + "/window_processing_times.csv";
        ofstream timing_ofs(timing_filename);
        if (timing_ofs) {
            timing_ofs << "WindowIndex,ProcessingTimeSeconds\n";
            timing_ofs << fixed << setprecision(4);
            for (size_t i = 0; i < window_processing_times.size(); ++i) {
                timing_ofs << i << "," << window_processing_times[i] << "\n";
            }
            timing_ofs.close();
            // LOGF("Window processing times saved to: %s", timing_filename.c_str());
        } else {
            LOGF("Error: Could not open %s to save window timing.", timing_filename.c_str());
        }

        // Save Detailed Module Timing
        const string detailed_timing_filename = save_dir + "/module_timing_details.csv";
        ofstream detailed_timing_ofs(detailed_timing_filename);
        if (detailed_timing_ofs) {
            detailed_timing_ofs << "WindowIndex";
            for (const auto& module : module_names) detailed_timing_ofs << "," << module;
            detailed_timing_ofs << "\n";
            detailed_timing_ofs << fixed << setprecision(4);
            size_t num_timing_entries = !module_durations.empty() && module_durations.count(module_names[0]) ? module_durations.at(module_names[0]).size() : 0;
            // Use window_processing_times size as reference if module timings might be absent
             num_timing_entries = std::max(num_timing_entries, window_processing_times.size());

            for (size_t window_idx = 0; window_idx < num_timing_entries; ++window_idx) {
                detailed_timing_ofs << window_idx;
                for (const auto& module : module_names) {
                     if (module_durations.count(module) && window_idx < module_durations.at(module).size()) {
                         detailed_timing_ofs << "," << module_durations.at(module)[window_idx];
                     } else {
                         detailed_timing_ofs << ",NA";
                     }
                }
                detailed_timing_ofs << "\n";
            }
            // Summary Stats
            detailed_timing_ofs << "\nSummary Statistics (Seconds):\n";
            detailed_timing_ofs << "Module,Count,Min,Max,Mean,Total\n";
            for (const auto& module : module_names) {
                 if (module_durations.count(module)) {
                    const auto& times = module_durations.at(module);
                    if (!times.empty()) {
                        double min_time = *std::min_element(times.begin(), times.end());
                        double max_time = *std::max_element(times.begin(), times.end());
                        double total_time = std::accumulate(times.begin(), times.end(), 0.0);
                        double mean_time = total_time / times.size();
                        detailed_timing_ofs << module << "," << times.size() << "," << min_time << "," << max_time << "," << mean_time << "," << total_time << "\n";
                    } else detailed_timing_ofs << module << ",0,NA,NA,NA,NA\n";
                 } else detailed_timing_ofs << module << ",0,NA,NA,NA,NA\n";
            }
            detailed_timing_ofs.close();
            // LOGF("Detailed module timing info saved to: %s", detailed_timing_filename.c_str());
        } else {
             LOGF("Error: Could not open %s to save detailed module timing.", detailed_timing_filename.c_str());
        }

        // Save Window Results (Labels and Scores)
        if (!window_results.empty()) LOGF("Saving %zu window results to directory %s...", window_results.size(), save_dir.c_str());
        for (size_t i = 0; i < this->window_results.size(); ++i) {
            const string filename = save_dir + "/window_" + to_string(i) + "_results.bin";
            ofstream ofs(filename, ios::binary | ios::trunc);
            if (ofs) {
                try {
                    const auto& labels_ptr = this->window_results[i].first;
                    if (labels_ptr) {
                        const auto& labels = *labels_ptr;
                        const size_t label_size = labels.size();
                        ofs.write(reinterpret_cast<const char*>(&label_size), sizeof(label_size));
                        for (const bool label_val : labels) { char b = label_val; ofs.write(&b, sizeof(b)); }
                    } else { size_t s=0; ofs.write(reinterpret_cast<const char*>(&s), sizeof(s)); }

                    const auto& scores_ptr = this->window_results[i].second;
                     if (scores_ptr) {
                        const auto& scores = *scores_ptr;
                        const size_t score_size = scores.size();
                        ofs.write(reinterpret_cast<const char*>(&score_size), sizeof(score_size));
                        if (score_size > 0) ofs.write(reinterpret_cast<const char*>(scores.data()), score_size * sizeof(double));
                    } else { size_t s=0; ofs.write(reinterpret_cast<const char*>(&s), sizeof(s)); }

                    if (!ofs) LOGF("Error writing data to file: %s (Window %zu)", filename.c_str(), i);
                    // else LOGF("Window %zu results saved: %s", i, filename.c_str()); // Can be verbose

                } catch (const std::exception& e) {
                     LOGF("Exception writing window %zu results to %s: %s", i, filename.c_str(), e.what());
                }
                ofs.close();
            } else {
                LOGF("Error: Failed to open file for writing window %zu results: %s", i, filename.c_str());
            }
        }
        LOGF("Finished saving results for directory: %s", save_dir.c_str());
    }


private:
    // save_model remains the same
    void save_model(const shared_ptr<traffic_graph>& model, const string& path) const {
        // ... (implementation unchanged) ...
        string save_dir = ".";
        size_t last_slash = path.find_last_of("/\\");
        if (last_slash != string::npos) save_dir = path.substr(0, last_slash);

        if (!create_directory_if_not_exists(save_dir)) {
            LOGF("Error: Failed to create directory %s for saving model %s", save_dir.c_str(), path.c_str());
            return;
        }

        ofstream ofs(path, ios::binary | ios::trunc);
        if (!ofs) {
            LOGF("Error: Failed to open model file for writing: %s", path.c_str());
            return;
        }
        try {
            if (model) {
                model->serialize(ofs);
                LOGF("Model serialized successfully to: %s", path.c_str());
            } else {
                LOGF("Error: Attempted to save a null model pointer to %s", path.c_str());
            }
        } catch (const std::exception& e) {
             LOGF("Exception during model serialization to %s: %s", path.c_str(), e.what());
        }
        ofs.close();
    }

    // load_model remains the same
    shared_ptr<traffic_graph> load_model(const string& path) const {
        // ... (implementation unchanged) ...
        ifstream ifs(path, ios::binary);
        if (!ifs) {
            LOGF("Failed to open model file for reading: %s", path.c_str());
            return nullptr;
        }
        try {
            auto model = make_shared<traffic_graph>();
            if (model->deserialize(ifs)) {
                // LOGF("Model deserialized successfully from: %s", path.c_str()); // Can be verbose
                return model;
            } else {
                LOGF("Failed to deserialize model data from: %s", path.c_str());
                return nullptr;
            }
        } catch (const exception& e) {
            LOGF("Exception during model deserialization from %s: %s", path.c_str(), e.what());
            return nullptr;
        }
    }

    // apply_incremental_learning remains the same
    void apply_incremental_learning(const shared_ptr<traffic_graph>& current_model,
                                   const shared_ptr<traffic_graph>& previous_model) const {
        // ... (implementation unchanged) ...
        if (!previous_model || !previous_model->has_model()) {
            LOGF("No valid previous model provided for incremental learning step.");
            return;
        }
        if (!current_model) {
            LOGF("Error: Current model is null in apply_incremental_learning.");
            return;
        }
        // LOGF("Applying incremental update. Decay: %.3f, LR: %.3f", model_decay_factor, learning_rate); // Can be verbose
        current_model->apply_incremental_update(previous_model, model_decay_factor, learning_rate);
    }
};

} // namespace Hypervision
