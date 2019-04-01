#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

int main(int argc, char**argv)
{
	if(argc<2)
		return 1;
	// Тренировка умножения

    srand(time(0));
	cout << "topology: 2 4 1" << endl;
	for(int i = atoi(argv[1]); i >= 0; --i)
	{
	    double n1 = 0.1*(rand()%9+1);
		double n2 = 0.1*(rand()%9+1);
		double t = n1 * n2; 
		cout << "in: " << n1 << " " << n2 << " " << endl;
		cout << "out: " << t << endl; 
	}

}