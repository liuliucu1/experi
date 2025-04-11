#include "basic_dataset.hpp"


using namespace Hypervision;


void basic_dataset::do_dataset_construct(size_t multiplex) {
    __START_FTIMMER__

    if (p_parse_result == nullptr) {
        FATAL_ERROR("Parsed dataset not found.");
    }
    if (p_parse_train != nullptr || p_parse_test != nullptr || p_label != nullptr) {
        WARN("The construction of dataset has already be done.");
    }

    size_t line = ceil(p_parse_result->size() * train_ratio);
    p_parse_train = make_shared<decltype(p_parse_train)::element_type>
                        (p_parse_result->begin(), p_parse_result->begin() + line);
    p_parse_test = make_shared<decltype(p_parse_test)::element_type>
                        (p_parse_result->begin() + line, p_parse_result->end());
    LOGF("[Train set: %8ld packets]", p_parse_train->size());

    // Load labels from file instead of generating them
    ifstream _ifl(load_label_path);
    if (!_ifl.is_open()) {
        FATAL_ERROR("Cannot open label file.");
    }
    
    p_label = make_shared<decltype(p_label)::element_type>();
    string ll;
    _ifl >> ll;
    
    // Only use the portion of labels corresponding to the test set
    for (size_t i = line; i < p_parse_result->size() && (i - line) < ll.size(); i++) {
        p_label->push_back(ll[i] == '1');
    }
    _ifl.close();
    
    // Ensure label size matches test set size
    if (p_label->size() != p_parse_test->size()) {
        WARN("Label size does not match test set size. Adjusting...");
        if (p_label->size() < p_parse_test->size()) {
            // If we have fewer labels than test samples, fill the rest with false
            fill_n(back_inserter(*p_label), p_parse_test->size() - p_label->size(), false);
        } else if (p_label->size() > p_parse_test->size()) {
            // If we have more labels than test samples, truncate
            p_label->resize(p_parse_test->size());
        }
    }
    
    LOGF("p_label size: %lu", p_label->size());
    LOGF("p_parse_test size: %lu", p_parse_test->size());

    size_t num_malicious = count(p_label->begin(), p_label->end(), true);
    LOGF("[test  set: %8ld packets]", p_parse_test->size());
    LOGF("[%8ld benign (%4.2lf%%), %8ld malicious (%4.2lf%%)]", 
        p_parse_test->size() - num_malicious,
        100.0 * (p_parse_test->size() - num_malicious) /  p_parse_test->size(),
        num_malicious,
        100.0 * (num_malicious) /  p_parse_test->size());

    __STOP_FTIMMER__
    __PRINTF_EXE_TIME__
    
}


void basic_dataset::configure_via_json(const json & jin) {
    try {
        if (jin.count("train_ratio")) {
            train_ratio = static_cast<decltype(train_ratio)>(jin["train_ratio"]);
            if (train_ratio < -EPS) {
                FATAL_ERROR("Ratio of training data is lower than 0.");
            }
        }
        if (jin.count("attack_time_after")) {
            attack_time_after = static_cast<decltype(attack_time_after)>(jin["attack_time_after"]);
        }

        if (jin.count("data_path")) {
            load_data_path = static_cast<decltype(load_data_path)>(jin["data_path"]);
        }
        if (jin.count("label_path")) {
            load_label_path = static_cast<decltype(load_label_path)>(jin["label_path"]);
        }

        if (jin.count("attacker_src4") && jin["attacker_src4"].size() != 0) {
            if (p_attacker_src4 != nullptr) {
                WARN("Reconfigure attacker source IPv4 list");
            }
            p_attacker_src4 = make_shared<decltype(p_attacker_src4)::element_type>();
            const auto _ls = jin["attacker_src4"];
            for (const auto & _l: _ls) {
                p_attacker_src4->push_back(static_cast<string>(_l));
            }
        }
        if (jin.count("attacker_src6") && jin["attacker_src6"].size() != 0) {
            if (p_attacker_src6 != nullptr) {
                WARN("Reconfigure attacker source IPv6 list");
            }
            p_attacker_src6 = make_shared<decltype(p_attacker_src6)::element_type>();
            const auto _ls = jin["attacker_src6"];
            for (const auto & _l: _ls) {
                p_attacker_src6->push_back(static_cast<string>(_l));
            }
        }

        if (jin.count("attacker_dst4") && jin["attacker_dst4"].size() != 0) {
            if (p_attacker_dst4 != nullptr) {
                WARN("Reconfigure attacker destination IPv4 list");
            }
            p_attacker_dst4 = make_shared<decltype(p_attacker_dst4)::element_type>();
            const auto _ls = jin["attacker_dst4"];
            for (const auto & _l: _ls) {
                p_attacker_dst4->push_back(static_cast<string>(_l));
            }
        }
        if (jin.count("attacker_dst6") && jin["attacker_dst6"].size() != 0) {
            if (p_attacker_dst6 != nullptr) {
                WARN("Reconfigure attacker destination IPv6 list");
            }
            p_attacker_dst6 = make_shared<decltype(p_attacker_dst6)::element_type>();
            const auto _ls = jin["attacker_dst4"];
            for (const auto & _l: _ls) {
                p_attacker_dst6->push_back(static_cast<string>(_l));
            }
        }

        if (jin.count("attacker_srcdst4") && jin["attacker_srcdst4"].size() != 0) {
            if (p_attacker_srcdst4 != nullptr) {
                WARN("Reconfigure attacker source-destination IPv4 list");
            }
            p_attacker_srcdst4 = make_shared<decltype(p_attacker_srcdst4)::element_type>();
            const auto _ls = jin["attacker_srcdst4"];
            for (const auto & _l: _ls) {
                if (_l.size() != 2) {
                    FATAL_ERROR("Wrong configuration format.");
                }
                p_attacker_srcdst4->push_back({static_cast<string>(_l[0]), static_cast<string>(_l[1])});
            }
        }
        if (jin.count("attacker_srcdst6") && jin["attacker_srcdst6"].size() != 0) {
            if (p_attacker_srcdst6 != nullptr) {
                WARN("Reconfigure attacker source-destination IPv6 list");
            }
            p_attacker_srcdst6 = make_shared<decltype(p_attacker_srcdst6)::element_type>();
            const auto _ls = jin["attacker_srcdst6"];
            for (const auto & _l: _ls) {
                if (_l.size() != 2) {
                    FATAL_ERROR("Wrong configuration format.");
                }
                p_attacker_srcdst6->push_back({static_cast<string>(_l[0]), static_cast<string>(_l[1])});
            }
        }
        
    } catch(const exception & e) {
        FATAL_ERROR(e.what());
    }
    
}
