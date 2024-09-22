#include <iostream>
#include <string>
#include <map>

using namespace std;

int RockPaperScissor(string player1, string player2)
{   
    map<string, int> rcpmap;

    // Insert some values into the map
    rcpmap["rock"] = 0;
    rcpmap["paper"] = 1;
    rcpmap["scissor"] = 2;

    if ((rcpmap[player1] + 1)%3 == rcpmap[player2])
        cout << "Player 2 wins!\n";
    else if ((rcpmap[player2] + 1)%3 == rcpmap[player1])
        cout << "Player 1 wins!\n";
    else if (player1 == player2)
        cout << "Draw!\n";
    else
        cout << "invalid input\n";

    return 0;
}

int main()
{
    string values[3] = {"rock", "paper", "scissor"};

    for (string c1: values){
        for (string c2: values){
            cout << "P1:" <<c1 << " P2:" << c2 << " ---> ";
            RockPaperScissor(c1, c2);
        }
    }
    return 0;
}