#include <iostream>
#include <ctime>
#include "trigger_al_Ver1.0a.h"

using namespace std;

float * make_data(void);

int main(void){

    float * test_data = make_data();

    float * temp = new float[64000]{};

    // Trig_al a(4, 16000);
    Trig_al a(1.0);
    
    a.preprocess_progressing(test_data, temp, 64000);

    // bool temp_bool = a.trigger_boolean(test_data, 64000);

    // cout << temp_bool << endl;

    float * temp_std = new float[64000]{};

    cout << "hello, world~!!" << endl;
    
    a.standardization_process(test_data, temp_std, 64000);

    float max_val=0, min_val=0;

    for(int i=0;i<64000;i++){
        if(max_val<temp_std[i]){
            max_val=temp_std[i];
        }
        if(min_val>temp_std[i]){
            min_val=temp_std[i];
        }
    }

    cout << endl << endl << max_val << "\t" << min_val << endl;

    return 0;
}


float * make_data(void){
    srand(time(NULL));

    float * result = new float[64000]{};

    for(int i=8000; i < 28000; i++){
        result[i] = (rand()-16384)%100;
    }

    return result;
}