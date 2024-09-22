#include <iostream>
#include <string>
using namespace std;

struct Students
{
    string name;
    string email;
    string username;
    string research;
};

void CoutStudent(const Students &s)
{
    cout << "Name: " << s.name << endl;
    cout << "Email: " << s.email << endl;
    cout << "Username: " << s.username << endl;
    cout << "Research: " << s.research << endl;
}

int main()
{
    string student_list[] = {"Rupesh", "Ameya", "Miranda"};

    Students rupesh, ameya, miranda;

    rupesh.name = "Rupesh Kannan";
    rupesh.email = "rupesh@wisc.edu";
    rupesh.username = "rupeshknn";
    rupesh.research = "Quantum Computing";

    ameya.name = "Ameya Thete";
    ameya.email = "ameya@wisc.edu";
    ameya.username = "ameyat05";
    ameya.research = "HEP";

    miranda.name = "Miranda Gorsuch";
    miranda.email = "miranda@wisc.edu";
    miranda.username = "mirandag12";
    miranda.research = "Cosmology";

    CoutStudent(rupesh);
    return 0;
}
