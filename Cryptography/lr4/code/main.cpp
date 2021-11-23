#include <bits/stdc++.h>

using namespace std;

void solve(ifstream &source, ifstream& changed) {

    char c1;
    char c2;
    int count = 0;

    while (!source.eof() && !changed.eof()) {
        source.read(&c1, sizeof(c1));
        changed.read(&c2, sizeof(c2));

        for (int i = 0; i < sizeof(c1) * 8; i++) {
            if ((c1 & 1) != (c2 & 1)) count++;
            c1 >>= 1;
            c2 >>= 1;
        }
    }

    if (source.eof() != changed.eof()) {
        cout << "invalid size of files";
        return;
    }

    cout << "Different bits count: " << count;
}

int main() {
    ifstream source( "template.bin", std::ios::binary);
    ifstream changed( "out.bin", std::ios::binary);

    if (!source.is_open() || !changed.is_open()) {
        cout << "cannot open files";
        return 1;
    }

    solve(source, changed);

    source.close();
    changed.close();
    return 0;
}
