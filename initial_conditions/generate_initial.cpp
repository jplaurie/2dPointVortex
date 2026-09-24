#include "initial_condition.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

[[noreturn]] void usage(int status) {
    std::ostream &out = status == 0 ? std::cout : std::cerr;
    out << "Usage: point_vortex_initial --geometry GEOMETRY --case CASE --output FILE [options]\n"
           "\n"
           "Geometries: infinite, periodic_x, periodic, disk\n"
           "Cases:      random, single, pair, dipole, ring\n"
           "\n"
           "Options:\n"
           "  --count N                 Vortex count for random/ring (default 100)\n"
           "  --seed N                  Random seed (default 1234567)\n"
           "  --min-separation D        Reject pairs closer than D (default 0)\n"
           "  --circulation G           Absolute circulation (default 1)\n"
           "  --half-width L            Unbounded-direction random extent (default 1)\n"
           "  --box-length L            Periodic length (default 2)\n"
           "  --disk-radius R            Disk radius (default 1)\n"
           "  --ring-radius R            Ring radius; 0 selects an automatic value\n"
           "  --overwrite                Permit replacing FILE\n"
           "  --help                     Show this help\n";
    std::exit(status);
}

double parseDouble(const std::string &text, const char *option) {
    std::size_t parsed = 0;
    const double value = std::stod(text, &parsed);
    if (parsed != text.size() || !std::isfinite(value))
        throw std::invalid_argument(std::string(option) + " expects a finite number");
    return value;
}

std::uint64_t parseUnsigned(const std::string &text, const char *option) {
    if (text.empty() || text.front() == '-')
        throw std::invalid_argument(std::string(option) + " expects a non-negative integer");
    std::size_t parsed = 0;
    const auto value = std::stoull(text, &parsed);
    if (parsed != text.size())
        throw std::invalid_argument(std::string(option) + " expects an integer");
    return value;
}

std::string nextArgument(int &index, int argc, char **argv) {
    if (++index == argc)
        throw std::invalid_argument(std::string("missing value after ") + argv[index - 1]);
    return argv[index];
}

InitialGeometry parseGeometry(const std::string &value) {
    if (value == "infinite")
        return InitialGeometry::infinite;
    if (value == "periodic_x")
        return InitialGeometry::periodic_x;
    if (value == "periodic")
        return InitialGeometry::periodic;
    if (value == "disk")
        return InitialGeometry::disk;
    throw std::invalid_argument("--geometry must be infinite, periodic_x, periodic, or disk");
}

InitialPattern parsePattern(const std::string &value) {
    if (value == "random")
        return InitialPattern::random;
    if (value == "single")
        return InitialPattern::single;
    if (value == "pair")
        return InitialPattern::corotatingPair;
    if (value == "dipole")
        return InitialPattern::dipole;
    if (value == "ring")
        return InitialPattern::ring;
    throw std::invalid_argument("--case must be random, single, pair, dipole, or ring");
}

} // namespace

int main(int argc, char **argv) {
    try {
        InitialConditionOptions options;
        std::string outputFile;
        bool geometrySpecified = false;
        bool overwrite = false;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--help")
                usage(0);
            if (argument == "--geometry") {
                options.geometry = parseGeometry(nextArgument(i, argc, argv));
                geometrySpecified = true;
            } else if (argument == "--case")
                options.pattern = parsePattern(nextArgument(i, argc, argv));
            else if (argument == "--output")
                outputFile = nextArgument(i, argc, argv);
            else if (argument == "--count")
                options.count = parseUnsigned(nextArgument(i, argc, argv), "--count");
            else if (argument == "--seed")
                options.seed = parseUnsigned(nextArgument(i, argc, argv), "--seed");
            else if (argument == "--min-separation")
                options.minimumSeparation =
                    parseDouble(nextArgument(i, argc, argv), "--min-separation");
            else if (argument == "--circulation")
                options.circulationMagnitude =
                    parseDouble(nextArgument(i, argc, argv), "--circulation");
            else if (argument == "--half-width")
                options.infiniteHalfWidth =
                    parseDouble(nextArgument(i, argc, argv), "--half-width");
            else if (argument == "--box-length")
                options.boxLength = parseDouble(nextArgument(i, argc, argv), "--box-length");
            else if (argument == "--disk-radius")
                options.diskRadius = parseDouble(nextArgument(i, argc, argv), "--disk-radius");
            else if (argument == "--ring-radius")
                options.ringRadius = parseDouble(nextArgument(i, argc, argv), "--ring-radius");
            else if (argument == "--overwrite")
                overwrite = true;
            else
                throw std::invalid_argument("unknown option: " + argument);
        }
        if (!geometrySpecified)
            throw std::invalid_argument("--geometry is required");
        if (outputFile.empty())
            throw std::invalid_argument("--output is required");

        const VortexSystem vortices = generateInitialCondition(options);
        writeInitialCondition(outputFile, vortices, options, overwrite);
        const double separation = minimumVortexSeparation(vortices, options);
        std::cout << "wrote " << vortices.size() << " vortices to " << outputFile
                  << "\ngeometry=" << toString(options.geometry)
                  << " case=" << toString(options.pattern) << " seed=" << options.seed
                  << " minimum_separation=" << separation << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
