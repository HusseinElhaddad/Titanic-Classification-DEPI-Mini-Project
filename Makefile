CXX = g++
CXXFLAGS = -std=c++17 -O2

# Main GCD permutation solution
gcd_permutation: gcd_permutation.cpp
	$(CXX) $(CXXFLAGS) -o gcd_permutation gcd_permutation.cpp

# Verification tools
verify_solution: verify_solution.cpp
	$(CXX) $(CXXFLAGS) -o verify_solution verify_solution.cpp

test_expected: test_expected.cpp
	$(CXX) $(CXXFLAGS) -o test_expected test_expected.cpp

# Build all
all: gcd_permutation verify_solution test_expected

# Test the solution
test: gcd_permutation
	./gcd_permutation < test_input.txt

# Extended test
test_extended: gcd_permutation
	./gcd_permutation < extended_test_input.txt

# Verify our solutions
verify: test_expected
	./test_expected

# Clean compiled files
clean:
	rm -f gcd_permutation verify_solution test_expected

.PHONY: all test test_extended verify clean