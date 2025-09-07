# GCD Permutation Problem Solution

## Problem Statement

Given a permutation `p` of size `n`, find a permutation `q` of size `n` such that `GCD(pi+qi, pi+1+qi+1) >= 3` for all `1 <= i < n`.

In other words, the greatest common divisor of the sum of any two adjacent positions should be at least 3.

## Algorithm

The solution uses multiple construction strategies:

1. **Reverse Order**: Try `q = [n, n-1, n-2, ..., 1]`
2. **Cyclic Shifts**: Try different cyclic permutations
3. **Brute Force**: For small `n <= 8`, try all permutations
4. **Systematic Search**: Try various shift patterns

## Implementation

The main algorithm is in `gcd_permutation.cpp`:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int gcd(int a, int b);
bool isValidPermutation(vector<int>& p, vector<int>& q);
void solve();
```

## Usage

1. Compile the program:
```bash
g++ -o gcd_permutation gcd_permutation.cpp
```

2. Run with input:
```bash
./gcd_permutation < test_input.txt
```

## Test Cases

### Input Format
```
t                    # Number of test cases
n                    # Size of permutation
p1 p2 ... pn        # The permutation p
```

### Example
```
3
3
1 3 2
5
5 1 2 4 3
7
6 7 1 5 4 3 2
```

### Output
```
2 3 1
1 2 4 5 3
1 7 6 2 3 4 5
```

## Verification

The solution includes verification tools:

- `verify_solution.cpp`: Verifies that a given solution satisfies the GCD constraint
- `test_expected.cpp`: Tests specific permutation pairs

## Complexity

- **Time**: O(n! * n) in worst case (brute force for small n), O(n) for most practical cases
- **Space**: O(n) for storing permutations

## Notes

- The problem guarantees that a solution always exists
- Multiple valid solutions may exist; any valid one is acceptable
- The algorithm prioritizes simple constructions before falling back to brute force