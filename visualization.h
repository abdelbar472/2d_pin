#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <vector>
#include <string>

// Define the PlacedPiece struct
struct PlacedPiece {
    int id;
    int x;
    int y;
    int length;
    int width;
    bool rotated;

    PlacedPiece(int id, int x, int y, int l, int w, bool r)
            : id(id), x(x), y(y), length(l), width(w), rotated(r) {}
};

// Define the Plank class
class Plank {
public:
    std::vector<PlacedPiece> pieces;
    int usedArea;

    Plank() : usedArea(0) {}

    double getEfficiency() const;
};

// Generate SVG file visualizing the cutting diagram
void generateSVG(const std::vector<Plank>& planks, const std::string& filename);

#endif // VISUALIZATION_H