# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Iinclude

# Source files
SRC := main.cpp \
       src/data_utils.cpp \
       src/linear_regression.cpp \
       src/metrics.cpp

# Output executable
TARGET := linear_regression

# Default target
all: $(TARGET)

# Link the object files to create the binary
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Clean up build files
clean:
	rm -f $(TARGET)

.PHONY: all clean
