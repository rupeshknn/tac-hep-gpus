#include <iostream>
using namespace std;

// passed by reference
void swaper(int &a, int &b)
{
    int c;
    c = a;
    a = b;
    b = c;
}

int main()
{
    int i;
    int A[10] = {0,1,2,3,4,5,6,7,8,9};
    int B[10] = {10,11,12,13,14,15,16,17,18,19};

    cout << "Before\n";
    cout << "A = [";
    for(int idx : A)
        cout << idx << ",";
    cout << "]\n";

    cout << "B = [";
    for(int idx : B)
        cout << idx << ",";
    cout << "]\n";

    for (i = 0; i <= 9; i++)
    {
        swaper(A[i], B[i]);
    }
    
    cout << "After\n";
    cout << "A = [";
    for(int idx : A)
        cout << idx << ",";
    cout << "]\n";

    cout << "B = [";
    for(int idx : B)
        cout << idx << ",";
    cout << "]\n";
}