#include <iostream>
#include <vector>
using namespace std;

int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

bool verify_expected_solution(vector<int>& p, vector<int>& q) {
    int n = p.size();
    
    cout << "Verification for p = ";
    for (int i = 0; i < n; i++) {
        cout << p[i] << " ";
    }
    cout << "and q = ";
    for (int i = 0; i < n; i++) {
        cout << q[i] << " ";
    }
    cout << endl;
    
    for (int i = 0; i < n - 1; i++) {
        int sum1 = p[i] + q[i];
        int sum2 = p[i + 1] + q[i + 1];
        int g = gcd(sum1, sum2);
        
        cout << "GCD(" << sum1 << ", " << sum2 << ") = " << g << endl;
        
        if (g < 3) {
            cout << "FAILED: GCD is " << g << " which is less than 3" << endl;
            return false;
        }
    }
    
    cout << "PASSED: All GCD values are >= 3" << endl;
    return true;
}

int main() {
    // Test our updated solutions
    
    // Test case 1: p = [1, 3, 2], our q = [2, 3, 1]
    vector<int> p1 = {1, 3, 2};
    vector<int> q1 = {2, 3, 1};
    cout << "=== Test Case 1 (Our Solution) ===" << endl;
    verify_expected_solution(p1, q1);
    cout << endl;
    
    // Test case 2: p = [5, 1, 2, 4, 3], our q = [1, 2, 4, 5, 3]
    vector<int> p2 = {5, 1, 2, 4, 3};
    vector<int> q2 = {1, 2, 4, 5, 3};
    cout << "=== Test Case 2 (Our Solution) ===" << endl;
    verify_expected_solution(p2, q2);
    cout << endl;
    
    // Test case 3: p = [6, 7, 1, 5, 4, 3, 2], our q = [1, 7, 6, 2, 3, 4, 5]
    vector<int> p3 = {6, 7, 1, 5, 4, 3, 2};
    vector<int> q3 = {1, 7, 6, 2, 3, 4, 5};
    cout << "=== Test Case 3 (Our Solution) ===" << endl;
    verify_expected_solution(p3, q3);
    
    return 0;
}