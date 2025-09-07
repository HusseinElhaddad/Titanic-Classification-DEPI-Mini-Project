#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

bool isValidPermutation(vector<int>& p, vector<int>& q) {
    int n = p.size();
    for (int i = 0; i < n - 1; i++) {
        int sum1 = p[i] + q[i];
        int sum2 = p[i + 1] + q[i + 1];
        if (gcd(sum1, sum2) < 3) {
            return false;
        }
    }
    return true;
}

void solve() {
    int n;
    cin >> n;
    vector<int> p(n);
    
    for (int i = 0; i < n; i++) {
        cin >> p[i];
    }
    
    vector<int> q(n);
    
    // Strategy: Use a construction that works well in practice
    // Based on analysis of the problem, a good construction is often
    // to arrange the permutation to create sums that share common factors
    
    // Construction 1: Try reverse order
    for (int i = 0; i < n; i++) {
        q[i] = n - i;
    }
    
    if (isValidPermutation(p, q)) {
        // Output the permutation q
        for (int i = 0; i < n; i++) {
            cout << q[i];
            if (i < n - 1) cout << " ";
        }
        cout << endl;
        return;
    }
    
    // Construction 2: Try a pattern that often works
    // Use the fact that we want GCD >= 3, so try to make many sums divisible by 3
    for (int i = 0; i < n; i++) {
        q[i] = ((n - i - 1 + (n / 2)) % n) + 1;
    }
    
    if (isValidPermutation(p, q)) {
        // Output the permutation q
        for (int i = 0; i < n; i++) {
            cout << q[i];
            if (i < n - 1) cout << " ";
        }
        cout << endl;
        return;
    }
    
    // Construction 3: For small n, use brute force
    if (n <= 8) {
        for (int i = 0; i < n; i++) {
            q[i] = i + 1;
        }
        
        do {
            if (isValidPermutation(p, q)) {
                // Output the permutation q
                for (int i = 0; i < n; i++) {
                    cout << q[i];
                    if (i < n - 1) cout << " ";
                }
                cout << endl;
                return;
            }
        } while (next_permutation(q.begin(), q.end()));
    }
    
    // Fallback: If nothing works, try a systematic approach
    // This should always find a solution as the problem guarantees one exists
    for (int shift = 1; shift < n; shift++) {
        for (int i = 0; i < n; i++) {
            q[i] = (i + shift) % n + 1;
        }
        if (isValidPermutation(p, q)) {
            // Output the permutation q
            for (int i = 0; i < n; i++) {
                cout << q[i];
                if (i < n - 1) cout << " ";
            }
            cout << endl;
            return;
        }
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int t;
    cin >> t;
    
    while (t--) {
        solve();
    }
    
    return 0;
}