#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <fstream>
#include <iomanip>
#include <string>
#include <tuple>
#include <numeric>
#include <limits>
#include <chrono>
#include <unordered_set>
#include <sstream>

// Default plank size
const int PLANK_LENGTH = 240;
const int PLANK_WIDTH = 120;

struct Item {
    int length, width, quantity, id;
    std::string name;
    Item() : length(0), width(0), quantity(0), id(-1), name("") {}
    Item(int l, int w, int q, const std::string& n, int i = -1) : length(l), width(w), quantity(q), id(i), name(n) {}
};

struct Plank {
    int length, width;
    std::vector<std::tuple<int, int, int, int, std::string, bool>> items;

    Plank(int l = PLANK_LENGTH, int w = PLANK_WIDTH) : length(l), width(w) {}

    bool can_place(int x, int y, int l, int w) const {
        if (x + l > length || y + w > width) return false;
        for (const auto& item : items) {
            int px = std::get<0>(item);
            int py = std::get<1>(item);
            int pl = std::get<2>(item);
            int pw = std::get<3>(item);
            if (!(x + l <= px || px + pl <= x || y + w <= py || py + pw <= y)) {
                return false;
            }
        }
        return true;
    }

    bool place_item(int x, int y, const Item& item, bool rotated) {
        int l = rotated ? item.width : item.length;
        int w = rotated ? item.length : item.width;
        if (can_place(x, y, l, w)) {
            items.emplace_back(x, y, l, w, item.name, rotated);
            return true;
        }
        return false;
    }

    int get_used_area() const {
        int area = 0;
        for (const auto& item : items) {
            area += std::get<2>(item) * std::get<3>(item);
        }
        return area;
    }
};

// Pack items in the exact order provided
std::vector<Plank> pack_in_order(const std::vector<Item>& items, int plank_length = PLANK_LENGTH, int plank_width = PLANK_WIDTH, int step = 5) {
    std::vector<Plank> planks;

    for (const auto& item : items) {
        bool placed = false;

        // Try to place in existing planks
        for (auto& plank : planks) {
            // Try normal orientation
            for (int y = 0; y <= plank.width - item.width && !placed; y += step) {
                for (int x = 0; x <= plank.length - item.length && !placed; x += step) {
                    if (plank.place_item(x, y, item, false)) {
                        placed = true;
                        break;
                    }
                }
            }
            // Try rotated orientation
            if (!placed) {
                for (int y = 0; y <= plank.width - item.length && !placed; y += step) {
                    for (int x = 0; x <= plank.length - item.width && !placed; x += step) {
                        if (plank.place_item(x, y, item, true)) {
                            placed = true;
                            break;
                        }
                    }
                }
            }
            if (placed) break;
        }

        // If not placed, create new plank
        if (!placed) {
            Plank new_plank(plank_length, plank_width);
            if (new_plank.place_item(0, 0, item, false) || new_plank.place_item(0, 0, item, true)) {
                planks.push_back(new_plank);
            } else {
                throw std::runtime_error("Item " + item.name + " too large for plank.");
            }
        }
    }
    return planks;
}

// Optimized greedy packing baseline (Sorts by area: First Fit Decreasing)
std::vector<Plank> greedy_packing(const std::vector<Item>& items, int plank_length = PLANK_LENGTH, int plank_width = PLANK_WIDTH, int step = 5) {
    std::vector<Item> individual_items;

    // Expand items by quantity
    for (const auto& item : items) {
        for (int i = 0; i < item.quantity; ++i) {
            individual_items.emplace_back(item.length, item.width, 1, item.name + "_" + std::to_string(i+1));
        }
    }

    // Sort by area (largest first)
    std::sort(individual_items.begin(), individual_items.end(),
              [](const Item& a, const Item& b) {
                  return (a.length * a.width) > (b.length * b.width);
              });

    return pack_in_order(individual_items, plank_length, plank_width, step);
}

int fitness(const std::vector<Item>& order, int plank_length, int plank_width) {
    // GA uses pack_in_order to evaluate the specific permutation
    return -static_cast<int>(pack_in_order(order, plank_length, plank_width, 10).size()); // Larger step for speed
}

std::vector<Item> tournament_selection(const std::vector<std::vector<Item>>& population, int plank_length, int plank_width, int tournament_size = 3) {
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dis(0, population.size() - 1);

    int best_idx = dis(gen);
    int best_fitness = fitness(population[best_idx], plank_length, plank_width);

    for (int i = 1; i < tournament_size; ++i) {
        int idx = dis(gen);
        int current_fitness = fitness(population[idx], plank_length, plank_width);
        if (current_fitness > best_fitness) {
            best_fitness = current_fitness;
            best_idx = idx;
        }
    }
    return population[best_idx];
}

std::vector<Item> crossover(const std::vector<Item>& p1, const std::vector<Item>& p2) {
    if (p1.empty()) return p2;
    static thread_local std::mt19937 gen(std::random_device{}());

    int n = p1.size();
    std::uniform_int_distribution<> dis(0, n - 1);
    int start = dis(gen);
    int end = dis(gen);
    if (start > end) std::swap(start, end);

    std::vector<Item> child(n);
    std::unordered_set<int> in_child;

    // Copy segment from p1
    for (int i = start; i <= end; ++i) {
        child[i] = p1[i];
        in_child.insert(p1[i].id);
    }

    // Fill remaining from p2
    int p2_idx = 0;
    for (int i = 0; i < n; ++i) {
        if (i >= start && i <= end) continue;

        while (p2_idx < n && in_child.count(p2[p2_idx].id)) {
            p2_idx++;
        }

        if (p2_idx < n) {
            child[i] = p2[p2_idx];
            in_child.insert(p2[p2_idx].id);
            p2_idx++;
        }
    }
    return child;
}

void mutate(std::vector<Item>& order) {
    static thread_local std::mt19937 gen(std::random_device{}());
    if (order.size() < 2) return;

    std::uniform_int_distribution<> dis(0, order.size() - 1);
    std::uniform_real_distribution<double> prob(0.0, 1.0);

    // Swap two random positions
    if (prob(gen) < 0.8) {
        int i = dis(gen);
        int j = dis(gen);
        std::swap(order[i], order[j]);
    }

    // Rotate an item
    if (prob(gen) < 0.2) {
        int idx = dis(gen);
        std::swap(order[idx].length, order[idx].width);
    }
}

std::vector<Plank> genetic_algorithm(const std::vector<Item>& items, int plank_length = PLANK_LENGTH, int plank_width = PLANK_WIDTH, int pop_size = 50, int generations = 100, double mutation_rate = 0.15, bool quiet = false) {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (!quiet) std::cout << "Initializing GA (Pop: " << pop_size << ", Gen: " << generations << ")..." << std::endl;

    // Expand items and assign unique IDs for GA tracking
    std::vector<Item> base_items;
    int unique_id = 0;
    for (const auto& item : items) {
        for (int i = 0; i < item.quantity; ++i) {
            base_items.emplace_back(item.length, item.width, 1, item.name + "_" + std::to_string(i+1), unique_id++);
        }
    }

    if (!quiet) std::cout << "Total individual items: " << base_items.size() << std::endl;

    // Initialize population
    std::random_device rd;
    std::mt19937 rng(rd());
    std::vector<std::vector<Item>> population(pop_size, base_items);

    for (auto& individual : population) {
        std::shuffle(individual.begin(), individual.end(), rng);
    }

    std::vector<Item> best_solution = population[0];
    int best_fitness = fitness(best_solution, plank_length, plank_width);

    if (!quiet) std::cout << "Initial best fitness: " << -best_fitness << " planks" << std::endl;

    // Evolution
    for (int gen = 0; gen < generations; ++gen) {
        auto gen_start = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<Item>> new_population;
        new_population.reserve(pop_size);
        new_population.push_back(best_solution); // Elitism

        // Generate new population
        while (new_population.size() < static_cast<size_t>(pop_size)) {
            auto p1 = tournament_selection(population, plank_length, plank_width);
            auto p2 = tournament_selection(population, plank_length, plank_width);
            auto child = crossover(p1, p2);

            std::uniform_real_distribution<double> mut_prob(0.0, 1.0);
            if (mut_prob(rng) < mutation_rate) {
                mutate(child);
            }
            new_population.push_back(std::move(child));
        }

        population = std::move(new_population);

        // Update best solution
        for (const auto& individual : population) {
            int current_fitness = fitness(individual, plank_length, plank_width);
            if (current_fitness > best_fitness) {
                best_fitness = current_fitness;
                best_solution = individual;
            }
        }

        auto gen_end = std::chrono::high_resolution_clock::now();
        auto gen_time = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start);

        if (!quiet && (gen % 5 == 0 || gen == generations - 1)) {
            std::cout << "Gen " << std::setw(2) << gen << ": " << -best_fitness
                      << " planks (" << gen_time.count() << "ms)" << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    if (!quiet) std::cout << "GA completed in " << total_time.count() << "ms" << std::endl;

    return pack_in_order(best_solution, plank_length, plank_width);
}

void print_results(const std::vector<Plank>& planks, const std::string& method) {
    std::cout << "\n=== " << method << " RESULTS ===" << std::endl;
    std::cout << "Planks used: " << planks.size() << std::endl;

    int total_used = 0, total_available = 0;
    for (size_t i = 0; i < planks.size(); ++i) {
        int used = planks[i].get_used_area();
        int available = planks[i].length * planks[i].width;
        total_used += used;
        total_available += available;

        std::cout << "Plank " << i + 1 << ": " << planks[i].items.size()
                  << " items, " << used << "/" << available << " cm² ("
                  << std::fixed << std::setprecision(1)
                  << (double)used/available*100 << "%)" << std::endl;
    }

    std::cout << "Overall efficiency: " << std::fixed << std::setprecision(1)
              << (double)total_used/total_available*100 << "%" << std::endl;
}

void run_json_mode() {
    std::string input_str;
    std::string line;
    while (std::getline(std::cin, line)) {
        input_str += line;
    }

    if (input_str.empty()) return;

    // Simple manual JSON parsing for the expected format
    // Format: {"algorithm": "...", "items": [{"name": "...", "length": 10, "width": 10, "quantity": 1}, ...]}

    std::vector<Item> items;
    std::string algorithm = "genetic";
    int plank_l = PLANK_LENGTH;
    int plank_w = PLANK_WIDTH;

    // Extract algorithm
    size_t alg_pos = input_str.find("\"algorithm\":");
    if (alg_pos != std::string::npos) {
        size_t start = input_str.find("\"", alg_pos + 12) + 1;
        size_t end = input_str.find("\"", start);
        algorithm = input_str.substr(start, end - start);
    }

    // Extract items
    size_t items_pos = input_str.find("\"items\":");
    if (items_pos != std::string::npos) {
        size_t list_start = input_str.find("[", items_pos);
        size_t list_end = input_str.find("]", list_start);
        std::string items_list = input_str.substr(list_start + 1, list_end - list_start - 1);

        size_t item_start = 0;
        while ((item_start = items_list.find("{", item_start)) != std::string::npos) {
            size_t item_end = items_list.find("}", item_start);
            std::string item_obj = items_list.substr(item_start + 1, item_end - item_start - 1);

            std::string name;
            int l = 0, w = 0, q = 0;

            size_t name_pos = item_obj.find("\"name\":");
            if (name_pos != std::string::npos) {
                size_t s = item_obj.find("\"", name_pos + 7) + 1;
                size_t e = item_obj.find("\"", s);
                name = item_obj.substr(s, e - s);
            }

            size_t l_pos = item_obj.find("\"length\":");
            if (l_pos != std::string::npos) l = std::stoi(item_obj.substr(l_pos + 9));

            size_t w_pos = item_obj.find("\"width\":");
            if (w_pos != std::string::npos) w = std::stoi(item_obj.substr(w_pos + 8));

            size_t q_pos = item_obj.find("\"quantity\":");
            if (q_pos != std::string::npos) q = std::stoi(item_obj.substr(q_pos + 11));

            items.emplace_back(l, w, q, name);
            item_start = item_end + 1;
        }
    }

    std::vector<Plank> result;
    if (algorithm == "greedy") {
        result = greedy_packing(items, plank_l, plank_w);
    } else {
        result = genetic_algorithm(items, plank_l, plank_w, 50, 100, 0.15, true);
    }

    // Output JSON result
    std::cout << "{\"planks\": [";
    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << "{\"dimensions\": {\"length\": " << result[i].length << ", \"width\": " << result[i].width << "}, \"items\": [";
        for (size_t j = 0; j < result[i].items.size(); ++j) {
            auto& item = result[i].items[j];
            std::cout << "{\"position\": {\"x\": " << std::get<0>(item) << ", \"y\": " << std::get<1>(item) << "}, ";
            std::cout << "\"size\": {\"length\": " << std::get<2>(item) << ", \"width\": " << std::get<3>(item) << "}, ";
            std::cout << "\"name\": \"" << std::get<4>(item) << "\", \"rotated\": " << (std::get<5>(item) ? "true" : "false") << "}";
            if (j < result[i].items.size() - 1) std::cout << ",";
        }
        std::cout << "]}";
        if (i < result.size() - 1) std::cout << ",";
    }
    std::cout << "]}" << std::endl;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--json") {
        run_json_mode();
        return 0;
    }
    std::cout << "=== 2D Bin Packing Optimizer ===" << std::endl;
    std::cout << "System: Intel i5-6440HQ @ 2.60GHz, 16GB RAM" << std::endl;
    std::cout << "Plank size: " << PLANK_LENGTH << "x" << PLANK_WIDTH << " cm\n" << std::endl;

    std::vector<Item> items = {
            Item(180, 80, 1, "Top"),
            Item(75, 80, 2, "Leg"),
            Item(75, 180, 1, "Back"),
            Item(75, 10, 1, "Front_Support")
    };

    // Display items
    std::cout << "Items to pack:" << std::endl;
    for (const auto& item : items) {
        std::cout << "  " << item.name << ": " << item.length << "x" << item.width
                  << " cm (qty: " << item.quantity << ")" << std::endl;
    }

    try {
        // Greedy algorithm
        std::cout << "\n--- Running Greedy Algorithm ---" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        auto greedy_result = greedy_packing(items);
        auto end = std::chrono::high_resolution_clock::now();
        auto greedy_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Greedy completed in " << greedy_time.count() << "ms" << std::endl;
        print_results(greedy_result, "GREEDY");

        // Genetic algorithm
        std::cout << "\n--- Running Genetic Algorithm ---" << std::endl;
        start = std::chrono::high_resolution_clock::now();
        auto ga_result = genetic_algorithm(items, PLANK_LENGTH, PLANK_WIDTH, 50, 100);
        end = std::chrono::high_resolution_clock::now();
        auto ga_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        print_results(ga_result, "GENETIC ALGORITHM");

        // Comparison
        std::cout << "\n=== PERFORMANCE COMPARISON ===" << std::endl;
        std::cout << "Greedy: " << greedy_result.size() << " planks in " << greedy_time.count() << "ms" << std::endl;
        std::cout << "GA: " << ga_result.size() << " planks in " << ga_time.count() << "ms" << std::endl;

        if (ga_result.size() < greedy_result.size()) {
            std::cout << "GA wins! Saved " << (greedy_result.size() - ga_result.size()) << " plank(s)" << std::endl;
        } else if (ga_result.size() > greedy_result.size()) {
            std::cout << "Greedy wins!" << std::endl;
        } else {
            std::cout << "Tie! Both methods used same number of planks" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}