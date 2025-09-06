#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <vector>
#include <string>

// Forward declare the Plank struct which will be defined in the main file
// but used in visualization.cpp
class Plank;

// Generate SVG file visualizing the cutting diagram
void generateSVG(const std::vector<Plank>& planks, const std::string& filename);

#endif // VISUALIZATION_H
