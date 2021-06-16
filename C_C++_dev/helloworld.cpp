#include <iostream>
#include <array>
#include <cstdlib>
#include <ctime>

using namespace std;

int main(void){

    cout << "hello, world~!!"<< endl;

    float * a;
    a = new float[10]{};

    printf("%d\n", sizeof(a));

    // for (int i=0;i<10;i++){
    //     // cout << i << " : "<< a[i] << endl;
    //     a[i] = float(i);
    //     cout << i << " : "<< a[i] << endl;
    // }

    cout << a[5] << endl;

    printf("%d\n", sizeof(a));

    cout << (sizeof(*a))/(sizeof (a[0])) << "  " << sizeof a[0] << endl;

    cout << (sizeof(a))/(sizeof (a[0])) << "  " << sizeof a[0] << endl;

    cout << (_msize(a))/(sizeof (a[0])) << "  " << sizeof a[0] << endl << endl << endl;

    float b[] = {1, 2, 3, 4, 5};

    cout << b << endl;

    // for (int i=0;i<10;i++){
    //     cout << i << " : "<< b[i] << endl;
    // }

    cout << (sizeof(b))/(sizeof (float)) << "  " << sizeof b[0] << endl;

    srand(time(NULL));
    cout << (rand()-16384)%1000 << endl;
    cout << (rand()-16384)%1000 << endl;
    cout << (rand()-16384)%1000 << endl;

    cout << float(10./3) << endl;


    cout << endl << endl;

    float * d = new float[10]{};

    cout << d << "\t" << d[0] << "\t" << *d << endl;
    cout << &d << "\t" << &d[0] << endl;


    int *ex_a = (int *)malloc(sizeof(int));

    *ex_a = 4;
    cout << endl << *ex_a << endl;
    cout << ex_a << endl;
    cout << &ex_a << endl;
    cout << sizeof(ex_a) << endl;

    cout << endl;

    float * ex_arr = (float *)malloc(sizeof(float)*10);
    
    int num = 100;
    for (int i=0; i<10; i++){
        ex_arr[i] = num;
        num++;
    }

    cout << ex_arr[100] << endl;
    cout << _msize(ex_arr) << endl;
    cout << sizeof(ex_arr) << endl;
    cout << sizeof(*ex_arr) << endl;
    cout << *ex_arr << endl;

    return 0;
}



