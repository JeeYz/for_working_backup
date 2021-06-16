#include <iostream>

using namespace std;


class Trig_al{

private:
    int record_sec = 4;
    int sample_rate = 16000;

    int front_buf = 4000;
    int tail_buf = 4000;

    float trigger_val = 1.0;
    
    int window_size = 512;
    int shift_size = 256;

    int full_size = 0;

    void print_variable(void);
    float cal_mean_value(float*);
    float * add_noise(float*);
    float * fit_full_size(float*);
    float * check_and_cut_data(float*);
    float * standardization_process(float*, int);

public:
    // Trig_al(int, int);
    Trig_al(float);
    ~Trig_al();

    void preprocess_progressing(float*, float*, int);
    bool trigger_boolean(float*, int);
    void change_record_sec(int);
    int check_the_input_data(float*);

};

// 기본틀
/*************************************************
@ func : 
@ param :
@ return : 
@ remarks : 
@ version : 
@ date : year, month, day
*************************************************/
