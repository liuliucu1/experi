#pragma once


#include "../common.hpp"
#include "edge_define.hpp"
#include "edge_constructor.hpp"
#include "../dataset_construct/basic_dataset.hpp"
#include "../flow_construct/explicit_constructor.hpp"

#include <mlpack/core.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <mlpack/core/data/scaler_methods/min_max_scaler.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

#include <z3++.h>

#include <fstream>
#include <iostream>

namespace Hypervision {

using long_edge_index = vector<size_t>;
using short_edge_index = vector<size_t>;
using addr_t = string;

// ... (rest of the code remains the same)

class traffic_graph {
private:

    using feature_t = vector<double>;
    using score_t = vector<double>;
    shared_ptr<vector<shared_ptr<short_edge> > > p_short_edge;
    shared_ptr<vector<shared_ptr<long_edge> > > p_long_edge;

    unordered_set<addr_t> vertex_set_long;
    unordered_set<addr_t> vertex_set_short_reduce;
    unordered_set<addr_t> vertex_set_short;
    unordered_map<addr_t, long_edge_index> long_edge_out;
    unordered_map<addr_t, long_edge_index> long_edge_in;
    unordered_map<addr_t, short_edge_index> short_edge_in;
    unordered_map<addr_t, short_edge_index> short_edge_in_agg;
    unordered_map<addr_t, short_edge_index> short_edge_out;
    unordered_map<addr_t, short_edge_index> short_edge_out_agg;

    shared_ptr<score_t> p_short_edge_score;
    shared_ptr<score_t> p_long_edge_score;
    shared_ptr<score_t> p_pkt_score;

    void parse_short_edge(void);
    void parse_long_edge(void);

    void dump_graph_statistic_long(void) const;
    void dump_graph_statistic_short(void) const;

    bool proto_cluster = true;
    uint32_t val_K = 10;
    double_t al = 0.1, bl = 1.0, cl = 0.5;
    double_t as = 0.1, bs = 1.0, cs = 0.5;
    double_t uc = 0.01, us = 0.001, ul = 0.05;
    uint32_t vc = 10,   vs = 20,    vl = 10;
    double_t select_ratio = 0.01;

    double_t offset_l = 0.0, offset_s = 0.0;
    
    // Model state for incremental learning
    arma::mat model_short_centroids;
    arma::mat model_long_centroids;
    bool has_trained_model = false;
    
    // Dynamic update parameters
    bool dynamic_updates_enabled = false;
    double update_threshold = 0.1;  // Threshold to trigger reclustering
    size_t min_packets_for_update = 50;  // Minimum number of packets to trigger an update
    
    // Packet queue for dynamic updates
    struct PacketQueueItem {
        shared_ptr<basic_packet> packet;
        double priority;
        size_t arrival_time;
        
        // Constructor
        PacketQueueItem(shared_ptr<basic_packet> p, double pri, size_t time) 
            : packet(p), priority(pri), arrival_time(time) {}
        
        // Comparison operator for priority queue
        bool operator<(const PacketQueueItem& other) const {
            return priority < other.priority;
        }
    };
    
    // Priority queue for incoming packets
    priority_queue<PacketQueueItem> packet_queue;
    
    // Queue statistics for queue theory application
    size_t queue_max_size = 10000;
    double avg_service_rate = 0.0;
    double avg_arrival_rate = 0.0;
    double avg_waiting_time = 0.0;
    size_t total_packets_processed = 0;
    size_t current_timestamp = 0;
    
    // Cache for dynamic updates
    unordered_map<addr_t, vector<size_t>> addr_to_new_edges;
    bool clusters_need_update = false;

public:

    traffic_graph(const decltype(p_short_edge) p_short_edge, const decltype(p_long_edge) p_long_edge):
        p_short_edge(p_short_edge), p_long_edge(p_long_edge) {}
        
    // Default constructor for model loading
    traffic_graph() : 
        p_short_edge(make_shared<vector<shared_ptr<short_edge>>>()),
        p_long_edge(make_shared<vector<shared_ptr<long_edge>>>()) {}

    traffic_graph(const traffic_graph &) = delete;
    traffic_graph & operator=(const traffic_graph &) = delete;
    virtual ~traffic_graph () {}
    
    // Public accessor for has_trained_model
    bool has_model() const {
        return has_trained_model;
    }
    
    // Enable or disable dynamic updates
    void set_dynamic_updates(bool enabled) {
        dynamic_updates_enabled = enabled;
    }
    
    // Add a packet to the priority queue for processing
    void add_packet_to_queue(shared_ptr<basic_packet> packet) {
        if (!dynamic_updates_enabled) return;
        
        current_timestamp++;
        
        // Calculate priority based on queue theory
        double priority = calculate_packet_priority(packet);
        
        // Add to priority queue
        packet_queue.emplace(packet, priority, current_timestamp);
        
        // Update queue statistics
        update_queue_statistics();
        
        // Process queue if it exceeds threshold
        if (packet_queue.size() >= min_packets_for_update) {
            process_packet_queue();
        }
    }
    
    // Process packets in the queue based on priority
    void process_packet_queue() {
        if (packet_queue.empty()) return;
        
        // Track the number of packets processed in this batch
        size_t processed_count = 0;
        vector<shared_ptr<basic_packet>> batch_packets;
        
        // Process packets in priority order
        while (!packet_queue.empty() && processed_count < min_packets_for_update) {
            auto item = packet_queue.top();
            packet_queue.pop();
            
            batch_packets.push_back(item.packet);
            processed_count++;
        }
        
        // Update flow and edge structures with new packets
        update_with_new_packets(batch_packets);
        
        // Update total processed count
        total_packets_processed += processed_count;
        
        // Update service rate
        if (avg_service_rate == 0.0) {
            avg_service_rate = processed_count;
        } else {
            avg_service_rate = 0.9 * avg_service_rate + 0.1 * processed_count;
        }
    }
    
    // Calculate packet priority based on queue theory
    double calculate_packet_priority(const shared_ptr<basic_packet>& packet) {
        // Base priority starts with 1.0
        double priority = 1.0;
        
        // 1. Apply Little's Law: L = λW (queue length = arrival rate * waiting time)
        // Higher waiting time should increase priority
        double waiting_factor = avg_waiting_time > 0 ? 
            (current_timestamp - packet_queue.size() / avg_service_rate) : 0;
        priority += 0.5 * waiting_factor;
        
        // 2. Consider utilization ratio (ρ = λ/μ)
        // If utilization is high, prioritize important packets
        double utilization = avg_arrival_rate / (avg_service_rate > 0 ? avg_service_rate : 1);
        
        // 3. Prioritize based on packet characteristics
        // Check if packet is TCP (using the packet type code)
        if ((packet->tp & ((1 << TCP_SYN) | (1 << TCP_ACK) | (1 << TCP_FIN) | (1 << TCP_RST))) != 0) {
            priority += 0.2;  // TCP packets might be more important for analysis
        }
        
        // 4. Prioritize based on packet length (larger packets might be more important)
        if (packet->len > 1000) {
            priority += 0.1;
        }
        
        // 5. Apply M/M/1 queue theory: higher priority for packets that reduce variance
        if (utilization < 0.9) {
            priority += 0.1 * (1 - utilization);  // Lower utilization means we can process more diverse packets
        } else {
            priority += 0.1;  // High utilization means we should focus on important packets
        }
        
        return priority;
    }
    
    // Update queue statistics based on new packet arrivals
    void update_queue_statistics() {
        // Update arrival rate using exponential moving average
        if (avg_arrival_rate == 0.0) {
            avg_arrival_rate = 1.0;
        } else {
            avg_arrival_rate = 0.9 * avg_arrival_rate + 0.1;
        }
        
        // Calculate average waiting time using Little's Law
        if (avg_service_rate > 0) {
            avg_waiting_time = packet_queue.size() / avg_service_rate;
        }
    }
    
    // Update clusters with new packet data
    void update_with_new_packets(const vector<shared_ptr<basic_packet>>& new_packets) {
        if (new_packets.empty()) return;
        
        // 1. Create flows from new packets
        auto new_packets_ptr = make_shared<vector<shared_ptr<basic_packet>>>();
        new_packets_ptr->insert(new_packets_ptr->end(), new_packets.begin(), new_packets.end());
        
        // Use the flow constructor from the project
        auto p_flow_constructor = make_shared<explicit_flow_constructor>(new_packets_ptr);
        p_flow_constructor->construct_flow();
        auto new_flows = p_flow_constructor->get_constructed_raw_flow();
        
        if (!new_flows) {
            LOGF("Failed to construct flows from new packets");
            return;
        }
        
        // 2. Create edges from new flows
        auto p_edge_constructor = make_shared<edge_constructor>(new_flows);
        p_edge_constructor->do_construct();
        shared_ptr<vector<shared_ptr<short_edge>>> new_short_edges;
        shared_ptr<vector<shared_ptr<long_edge>>> new_long_edges;
        tie(new_short_edges, new_long_edges) = p_edge_constructor->get_edge();
        
        // 3. Add new edges to existing edge collections
        size_t original_short_size = p_short_edge->size();
        size_t original_long_size = p_long_edge->size();
        
        p_short_edge->insert(p_short_edge->end(), new_short_edges->begin(), new_short_edges->end());
        p_long_edge->insert(p_long_edge->end(), new_long_edges->begin(), new_long_edges->end());
        
        // 4. Update edge indices
        for (size_t i = original_short_size; i < p_short_edge->size(); i++) {
            const auto& edge = p_short_edge->at(i);
            const auto& src = edge->get_src_str();
            const auto& dst = edge->get_dst_str();
            
            // Add to vertex sets
            vertex_set_short.insert(src);
            vertex_set_short.insert(dst);
            
            // Update edge indices
            if (short_edge_out.count(src) == 0) {
                short_edge_out[src] = {i};
            } else {
                short_edge_out[src].push_back(i);
            }
            
            if (short_edge_in.count(dst) == 0) {
                short_edge_in[dst] = {i};
            } else {
                short_edge_in[dst].push_back(i);
            }
            
            // Track new edges for incremental clustering
            addr_to_new_edges[src].push_back(i);
            addr_to_new_edges[dst].push_back(i);
        }
        
        for (size_t i = original_long_size; i < p_long_edge->size(); i++) {
            const auto& edge = p_long_edge->at(i);
            const auto& src = edge->get_src_str();
            const auto& dst = edge->get_dst_str();
            
            // Add to vertex sets
            vertex_set_long.insert(src);
            vertex_set_long.insert(dst);
            
            // Update edge indices
            if (long_edge_out.count(src) == 0) {
                long_edge_out[src] = {i};
            } else {
                long_edge_out[src].push_back(i);
            }
            
            if (long_edge_in.count(dst) == 0) {
                long_edge_in[dst] = {i};
            } else {
                long_edge_in[dst].push_back(i);
            }
        }
        
        // 5. Mark that clusters need to be updated
        clusters_need_update = true;
        
        // 6. If we have enough new data, update clusters incrementally
        if (new_short_edges->size() + new_long_edges->size() >= min_packets_for_update) {
            update_clusters_incrementally();
        }
    }
    
    // Update clusters incrementally with new edge data
    void update_clusters_incrementally() {
        if (!clusters_need_update || !has_trained_model) return;
        
        // 1. Identify affected components
        unordered_set<addr_t> affected_addresses;
        for (const auto& pair : addr_to_new_edges) {
            affected_addresses.insert(pair.first);
        }
        
        // 2. Create a component from affected addresses
        vector<addr_t> affected_component(affected_addresses.begin(), affected_addresses.end());
        
        // 3. Process just this component
        _proc_each_component(affected_component);
        
        // 4. Reset tracking data
        addr_to_new_edges.clear();
        clusters_need_update = false;
    }

    void parse_edge(void) {
        p_short_edge_score = make_shared<score_t>(p_short_edge->size());
        p_long_edge_score = make_shared<score_t>(p_long_edge->size());
        
        std::fill(p_short_edge_score->begin(), p_short_edge_score->end(), 0.0);
        std::fill(p_long_edge_score->begin(), p_long_edge_score->end(), 0.0);
        
        parse_short_edge();
        parse_long_edge();
    }

    void dump_graph_statistic(void) const {
        dump_graph_statistic_long();
        dump_graph_statistic_short();
    }

    constexpr static size_t huge_short_line = 50;
    constexpr static size_t huge_agg_short_line = 100;

    auto is_huge_short_edge(const addr_t addr) const -> bool;
    auto is_huge_agg_short_edge(const addr_t & addr) const -> bool;
    void dump_vertex_anomly(void) const;
    void dump_edge_anomly(void) const;


    using component = vector<vector<addr_t> >;
    auto connected_component() const -> shared_ptr<component>;

    auto component_select(const shared_ptr<component> p_com) const -> shared_ptr<vector<size_t>>;


private:
    auto __f_get_inout_degree(const addr_t addr) const -> pair<size_t, size_t>;
    auto _f_exeract_feature_short(const size_t index) const -> feature_t;
    auto _f_exeract_feature_long(const size_t index) const -> feature_t;
    auto _f_exeract_feature_short2(const size_t index) const -> feature_t;
    auto _f_exeract_feature_long2(const size_t index) const -> feature_t;
    auto __f_trans_armadillo_mat_T(const vector<feature_t> & mx) -> arma::mat;

    void _acquire_edge_index(const vector<addr_t> & addr_ls, 
                             unordered_set<size_t> & _long_index, unordered_set<size_t> & _short_index);
    auto _pre_process_short(const unordered_set<size_t> & _short_index,
                            arma::mat & dataset_short, arma::mat & centroids_short, arma::Row<size_t> & assignments_short) -> size_t;
    auto _pre_process_long(const unordered_set<size_t> & _long_index,
                           arma::mat & centroids_long, arma::Row<size_t> & assignments_long) -> size_t;

    void _process_short(const unordered_set<size_t> & _short_index, 
                        const arma::mat & dataset_short, const arma::mat & centroids_short, const arma::Row<size_t> & assignments_short);
    void _process_long(const unordered_set<size_t> & _long_index,
                       const arma::mat & centroids_long, const arma::Row<size_t> & assignments_long);
    void _proc_each_component(const vector<addr_t> & addr_ls);

public:
    auto graph_detect() {
        proc_components(connected_component());
    }

    auto proc_components(const shared_ptr<component> p_com) -> void;

    auto get_final_pkt_score(const shared_ptr<binary_label_t> p_label) -> const decltype(p_pkt_score);

    void config_via_json(const json & jin);
    
    // Serialization method to save model state
    bool serialize(std::ofstream& ofs) const {
        if (!ofs) {
            LOGF("Error: Output stream is not valid");
            return false;
        }
        
        try {
            // Save model parameters
            ofs.write(reinterpret_cast<const char*>(&proto_cluster), sizeof(proto_cluster));
            ofs.write(reinterpret_cast<const char*>(&val_K), sizeof(val_K));
            ofs.write(reinterpret_cast<const char*>(&al), sizeof(al));
            ofs.write(reinterpret_cast<const char*>(&bl), sizeof(bl));
            ofs.write(reinterpret_cast<const char*>(&cl), sizeof(cl));
            ofs.write(reinterpret_cast<const char*>(&as), sizeof(as));
            ofs.write(reinterpret_cast<const char*>(&bs), sizeof(bs));
            ofs.write(reinterpret_cast<const char*>(&cs), sizeof(cs));
            ofs.write(reinterpret_cast<const char*>(&uc), sizeof(uc));
            ofs.write(reinterpret_cast<const char*>(&us), sizeof(us));
            ofs.write(reinterpret_cast<const char*>(&ul), sizeof(ul));
            ofs.write(reinterpret_cast<const char*>(&vc), sizeof(vc));
            ofs.write(reinterpret_cast<const char*>(&vs), sizeof(vs));
            ofs.write(reinterpret_cast<const char*>(&vl), sizeof(vl));
            ofs.write(reinterpret_cast<const char*>(&select_ratio), sizeof(select_ratio));
            ofs.write(reinterpret_cast<const char*>(&offset_l), sizeof(offset_l));
            ofs.write(reinterpret_cast<const char*>(&offset_s), sizeof(offset_s));
            
            // Save has_trained_model flag
            bool has_model = (model_short_centroids.n_elem > 0 || model_long_centroids.n_elem > 0);
            ofs.write(reinterpret_cast<const char*>(&has_model), sizeof(has_model));
            
            // Save centroids if they exist
            if (has_model) {
                // Save short edge centroids
                size_t short_rows = model_short_centroids.n_rows;
                size_t short_cols = model_short_centroids.n_cols;
                ofs.write(reinterpret_cast<const char*>(&short_rows), sizeof(short_rows));
                ofs.write(reinterpret_cast<const char*>(&short_cols), sizeof(short_cols));
                
                if (short_rows > 0 && short_cols > 0) {
                    ofs.write(reinterpret_cast<const char*>(model_short_centroids.memptr()), 
                             short_rows * short_cols * sizeof(double));
                }
                
                // Save long edge centroids
                size_t long_rows = model_long_centroids.n_rows;
                size_t long_cols = model_long_centroids.n_cols;
                ofs.write(reinterpret_cast<const char*>(&long_rows), sizeof(long_rows));
                ofs.write(reinterpret_cast<const char*>(&long_cols), sizeof(long_cols));
                
                if (long_rows > 0 && long_cols > 0) {
                    ofs.write(reinterpret_cast<const char*>(model_long_centroids.memptr()), 
                             long_rows * long_cols * sizeof(double));
                }
            }
            
            return true;
        } catch (const std::exception& e) {
            LOGF("Error during serialization: %s", e.what());
            return false;
        }
    }
    
    // Deserialization method to load model state
    bool deserialize(std::ifstream& ifs) {
        if (!ifs) {
            LOGF("Error: Input stream is not valid");
            return false;
        }
        
        try {
            // Load model parameters
            ifs.read(reinterpret_cast<char*>(&proto_cluster), sizeof(proto_cluster));
            ifs.read(reinterpret_cast<char*>(&val_K), sizeof(val_K));
            ifs.read(reinterpret_cast<char*>(&al), sizeof(al));
            ifs.read(reinterpret_cast<char*>(&bl), sizeof(bl));
            ifs.read(reinterpret_cast<char*>(&cl), sizeof(cl));
            ifs.read(reinterpret_cast<char*>(&as), sizeof(as));
            ifs.read(reinterpret_cast<char*>(&bs), sizeof(bs));
            ifs.read(reinterpret_cast<char*>(&cs), sizeof(cs));
            ifs.read(reinterpret_cast<char*>(&uc), sizeof(uc));
            ifs.read(reinterpret_cast<char*>(&us), sizeof(us));
            ifs.read(reinterpret_cast<char*>(&ul), sizeof(ul));
            ifs.read(reinterpret_cast<char*>(&vc), sizeof(vc));
            ifs.read(reinterpret_cast<char*>(&vs), sizeof(vs));
            ifs.read(reinterpret_cast<char*>(&vl), sizeof(vl));
            ifs.read(reinterpret_cast<char*>(&select_ratio), sizeof(select_ratio));
            ifs.read(reinterpret_cast<char*>(&offset_l), sizeof(offset_l));
            ifs.read(reinterpret_cast<char*>(&offset_s), sizeof(offset_s));
            
            // Load has_trained_model flag
            bool has_model = false;
            ifs.read(reinterpret_cast<char*>(&has_model), sizeof(has_model));
            
            // Load centroids if they exist
            if (has_model) {
                // Load short edge centroids
                size_t short_rows = 0, short_cols = 0;
                ifs.read(reinterpret_cast<char*>(&short_rows), sizeof(short_rows));
                ifs.read(reinterpret_cast<char*>(&short_cols), sizeof(short_cols));
                
                if (short_rows > 0 && short_cols > 0) {
                    model_short_centroids.set_size(short_rows, short_cols);
                    ifs.read(reinterpret_cast<char*>(model_short_centroids.memptr()), 
                            short_rows * short_cols * sizeof(double));
                }
                
                // Load long edge centroids
                size_t long_rows = 0, long_cols = 0;
                ifs.read(reinterpret_cast<char*>(&long_rows), sizeof(long_rows));
                ifs.read(reinterpret_cast<char*>(&long_cols), sizeof(long_cols));
                
                if (long_rows > 0 && long_cols > 0) {
                    model_long_centroids.set_size(long_rows, long_cols);
                    ifs.read(reinterpret_cast<char*>(model_long_centroids.memptr()), 
                            long_rows * long_cols * sizeof(double));
                }
                
                has_trained_model = true;
            }
            
            return true;
        } catch (const std::exception& e) {
            LOGF("Error during deserialization: %s", e.what());
            return false;
        }
    }
    
    // Apply incremental learning by combining previous model with current data
    void apply_incremental_update(const shared_ptr<traffic_graph>& previous_model, 
                                 double decay_factor, double learning_rate) {
        if (!previous_model || !previous_model->has_model()) {
            LOGF("No previous model available for incremental learning");
            return;
        }
        
        // Store the previous model's centroids for use in clustering
        model_short_centroids = previous_model->model_short_centroids;
        model_long_centroids = previous_model->model_long_centroids;
        has_trained_model = true;
        
        // Apply the decay factor to the learning parameters
        // This allows the model to gradually adapt to new data
        uc = (1.0 - learning_rate) * uc + (learning_rate * previous_model->uc);
        us = (1.0 - learning_rate) * us + (learning_rate * previous_model->us);
        ul = (1.0 - learning_rate) * ul + (learning_rate * previous_model->ul);
        
        // Copy other relevant parameters from the previous model
        // with a weighted combination of old and new values
        al = (1.0 - decay_factor) * al + decay_factor * previous_model->al;
        bl = (1.0 - decay_factor) * bl + decay_factor * previous_model->bl;
        cl = (1.0 - decay_factor) * cl + decay_factor * previous_model->cl;
        as = (1.0 - decay_factor) * as + decay_factor * previous_model->as;
        bs = (1.0 - decay_factor) * bs + decay_factor * previous_model->bs;
        cs = (1.0 - decay_factor) * cs + decay_factor * previous_model->cs;
        
        LOGF("Applied incremental learning with decay factor %f", decay_factor);
    }
    
    // Method to update clustering with previous model's centroids
    void initialize_clustering_from_model(arma::mat& centroids_short, arma::mat& centroids_long) {
        if (has_trained_model) {
            if (model_short_centroids.n_elem > 0) {
                centroids_short = model_short_centroids;
                LOGF("Initialized short edge clustering from previous model");
            }
            
            if (model_long_centroids.n_elem > 0) {
                centroids_long = model_long_centroids;
                LOGF("Initialized long edge clustering from previous model");
            }
        }
    }

};

}
