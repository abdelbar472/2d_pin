#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include "main.cpp"

int main() {
    std::cout << "=== 2D Bin Packing Calculator ===" << std::endl;
    std::cout << "Wood Blank Size: 240 x 120 cm" << std::endl;
    std::cout << "Enter your pieces to cut:" << std::endl;
    
    std::vector<Item> items;
    std::string line;
    
    std::cout << "Format: length width quantity name" << std::endl;
    std::cout << "Example: 100 50 3 shelf" << std::endl;
    std::cout << "Enter 'done' when finished:" << std::endl;
    
    while (std::getline(std::cin, line) && line != "done") {
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        int length, width, quantity;
        std::string name;
        
        if (iss >> length >> width >> quantity >> name) {
            items.emplace_back(length, width, quantity, name);
            std::cout << "Added: " << name << " " << length << "x" << width 
                     << " (qty: " << quantity << ")" << std::endl;
        } else {
            std::cout << "Invalid format. Please try again." << std::endl;
        }
    }
    
    if (items.empty()) {
        std::cout << "No items entered. Using default furniture example." << std::endl;
        items = {
            Item(180, 80, 1, "Top"),
            Item(75, 80, 2, "Leg"),
            Item(75, 180, 1, "Back"),
            Item(75, 10, 1, "Front_Support")
        };
    }
    
    try {
        // Run both algorithms
        auto greedy_result = greedy_packing(items);
        auto ga_result = genetic_algorithm(items, PLANK_LENGTH, PLANK_WIDTH, 20, 25);
        
        print_results(greedy_result, "GREEDY");
        print_results(ga_result, "GENETIC ALGORITHM");
        
        std::cout << "\n=== RECOMMENDATION ===" << std::endl;
        if (ga_result.size() <= greedy_result.size()) {
            std::cout << "Use Genetic Algorithm result: " << ga_result.size() << " planks needed" << std::endl;
        } else {
            std::cout << "Use Greedy Algorithm result: " << greedy_result.size() << " planks needed" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}