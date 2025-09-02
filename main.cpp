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

// Default plank size
const int PLANK_LENGTH = 240;
const int PLANK_WIDTH = 120;

struct Item {
    int length, width, quantity;
    std::string name;
    Item() : length(0), width(0), quantity(0), name("") {}
    Item(int l, int w, int q, const std::string& n) : length(l), width(w), quantity(q), name(n) {}
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

// Optimized greedy packing with larger step size for faster execution
std::vector<Plank> greedy_packing(const std::vector<Item>& items, int plank_length = PLANK_LENGTH, int plank_width = PLANK_WIDTH, int step = 5) {
    std::vector<Plank> planks;
    std::vector<Item> sorted_items;

    // Expand items by quantity
    for (const auto& item : items) {
        for (int i = 0; i < item.quantity; ++i) {
            sorted_items.emplace_back(item.length, item.width, 1, item.name + "_" + std::to_string(i+1));
        }
    }

    // Sort by area (largest first)
    std::sort(sorted_items.begin(), sorted_items.end(),
              [](const Item& a, const Item& b) {
                  return (a.length * a.width) > (b.length * b.width);
              });

    for (const auto& item : sorted_items) {
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

int fitness(const std::vector<Item>& order, int plank_length, int plank_width) {
    return -static_cast<int>(greedy_packing(order, plank_length, plank_width, 10).size()); // Larger step for speed
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
    static thread_local std::mt19937 gen(std::random_device{}());
    if (p1.empty()) return p2;

    std::uniform_int_distribution<> dis(0, p1.size() - 1);
    int start = dis(gen);
    int end = dis(gen);
    if (start > end) std::swap(start, end);

    std::vector<Item> child = p1;
    // Simple crossover - just swap a segment
    for (int i = start; i <= end && i < p2.size(); ++i) {
        child[i] = p2[i];
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

std::vector<Plank> genetic_algorithm(const std::vector<Item>& items, int plank_length = PLANK_LENGTH, int plank_width = PLANK_WIDTH, int pop_size = 20, int generations = 20, double mutation_rate = 0.1) {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "Initializing GA (Pop: " << pop_size << ", Gen: " << generations << ")..." << std::endl;

    // Expand items
    std::vector<Item> base_items;
    for (const auto& item : items) {
        for (int i = 0; i < item.quantity; ++i) {
            base_items.emplace_back(item.length, item.width, 1, item.name + "_" + std::to_string(i+1));
        }
    }

    std::cout << "Total individual items: " << base_items.size() << std::endl;

    // Initialize population
    std::random_device rd;
    std::mt19937 rng(rd());
    std::vector<std::vector<Item>> population(pop_size, base_items);

    for (auto& individual : population) {
        std::shuffle(individual.begin(), individual.end(), rng);
    }

    std::vector<Item> best_solution = population[0];
    int best_fitness = fitness(best_solution, plank_length, plank_width);

    std::cout << "Initial best fitness: " << -best_fitness << " planks" << std::endl;

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

        if (gen % 5 == 0 || gen == generations - 1) {
            std::cout << "Gen " << std::setw(2) << gen << ": " << -best_fitness
                      << " planks (" << gen_time.count() << "ms)" << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "GA completed in " << total_time.count() << "ms" << std::endl;

    return greedy_packing(best_solution, plank_length, plank_width);
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

int main() {
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
        auto ga_result = genetic_algorithm(items, PLANK_LENGTH, PLANK_WIDTH, 15, 15); // Reduced for speed
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