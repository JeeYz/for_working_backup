#include <iostream>
#include "trigger_al_Ver1.0a.h"
#include <array>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;


// public method : 
/*************************************************
@ func : 생성자 및 소멸자
@ param :
@ return : 
@ remarks : full size의 정의
@ version : trigger algorithm Version 1.0 alpha
@ date : 2021. 6. 15
*************************************************/
// Trig_al::Trig_al(int rsec, int sr){
//         record_sec = rsec;
//         sample_rate = sr;
//         full_size = record_sec*sample_rate;
//     }

Trig_al::Trig_al(float threshold){
        trigger_val = threshold;
        full_size = 20000;
    }

Trig_al::~Trig_al(){

    }



// private method : print_variable
/*************************************************
@ func : 테스트를 위한 함수
@ param :
@ return : 
@ remarks : private 변수 출력
@ version : trigger algorithm Version 1.0 alpha
@ date : 2021. 6. 15
*************************************************/
void Trig_al::print_variable(void){
    cout << record_sec << endl;
    cout << sample_rate << endl;
    return;
} 




// public method : trigger_boolean
/*************************************************
@ func : 트리거에 걸리는지 않는지 판단하는 함수
@ param : 신호 데이터 pointer value
@ return : boolean 값
@ remarks : 레어 데이터를 받아 표준화를 진행
            이 함수는 신호 리시브 스레드에서 사용
@ version : trigger algorithm Version 1.0 alpha
@ date : 2021. 6. 15
*************************************************/
bool Trig_al::trigger_boolean(float* data_arr, int data_size){

    float * temp = new float[full_size]{};
    // 이 부분이 유니티에서 실행 안될 수 있음.
    // _msize 함수 사용 때문에
    // 이유 :
    // 타 언어와 데이터 type과 value만 공유할 뿐
    // 내장 함수가 호환 안될 수도 있음

    // size_t len_data = (_msize(data_arr))/sizeof(float);
    int len_data = data_size;

    memcpy(temp, data_arr, sizeof(float)*len_data);

    float * temp_out = new float[len_data]{};
    
    standardization_process(temp, temp_out, len_data);

    const float mean_data = cal_mean_value(temp_out);

    if (mean_data > trigger_val){
        return true;
    }
    else{
        return false;
    }

    delete[] temp_out;
    delete[] temp;
}




// private method : cal_mean_value
/*************************************************
@ func : 입력 파라미터의 평균값을 계산
@ param : float 형태의 array pointer
@ return : 평균값 float
@ remarks : 마찬가지로 _msize 함수 문제가 발생할 수 있음
@ version : trigger algorithm Version 1.0 alpha
@ date : 2021. 6. 15
*************************************************/
float Trig_al::cal_mean_value(float * data_arr){

    size_t length_of_float = (_msize(data_arr))/sizeof data_arr[0];
    float fsum = 0;
    float mean_data = 0;

    for(int i=0;i < length_of_float; i++){

        if (data_arr[i]>=0){
            fsum = fsum+data_arr[i];    
        }
        else{
            fsum = fsum-data_arr[i];
        }

    }

    float result = fsum/length_of_float;

    return result;
}




// private method : add_noise
/*************************************************
@ func : 입력된 데이터에 미세 노이즈 추가
@ param : float 형태의 배열 pointer
@ return : 노이즈가 추가된 배열 float pointer
@ remarks : 
@ version : trigger algorithm Version 1.0 alpha
@ date : 2021. 6. 15
*************************************************/
float *Trig_al::add_noise(float * data_arr){

    srand(time(NULL));
    // _msize 함수 사용
    size_t len_data = (_msize(data_arr))/sizeof data_arr[0];

    for (int i=0; i<len_data; i++){
        data_arr[i] = data_arr[i]+((rand()-16384)%1000)*0.00001;
    }

    return data_arr;
}




// private method : fit_full_size
/*************************************************
@ func : 신호를 full size로 패딩
@ param : float 형태의 배열의 pointer
@ return : 패딩 처리된 신호 float pointer
@ remarks : 
@ version : trigger algorithm Version 1.0 alpha
@ date : 2021. 6. 15.
*************************************************/
float * Trig_al::fit_full_size(float * data_arr){

    size_t len_data = (_msize(data_arr))/sizeof data_arr[0];
    float * result = new float[record_sec*sample_rate]{};

    for (int i=0; i<len_data; i++){
        result[i] = data_arr[i];
    }

    delete[] data_arr;
    return result;
}




// private method : standardization_process
/*************************************************
@ func : 입력 신호를 표준화
@ param :   float pointer 형태의 배열
            입력 데이터의 사이즈 int
@ return : 표준화된 결과 데이터 float pointer
@ remarks : 
@ version : trigger algorithm Version 1.0 alpha
@ date : 2021. 6. 15.
@ the last update : 2021. 6.17
*************************************************/
void Trig_al::standardization_process(float* data_arr, float* result, int size_of_data){

    cout << "check point : 0" << endl;

    // size_t len_data = (_msize(data_arr))/sizeof data_arr[0];
    int len_data = size_of_data;

    cout << "check point : 1" << endl;

    float max_val = 0;
    float min_val = 0;

    float sum = 0;
    float var_mean = 0;
    float mean_val = 0;
    float std_val = 0;


    for (int i=0; i<len_data; i++){
        if (max_val < data_arr[i]){
            max_val = data_arr[i];
        }
        if (min_val > data_arr[i]){
            min_val = data_arr[i];
        }
        sum = sum + data_arr[i];
    }

    mean_val = sum / len_data;

    // cout << mean_val << endl;

    float temp_sum = 0;

    for (int i=0; i<len_data; i++){
        temp_sum = temp_sum+(data_arr[i]-mean_val)*(data_arr[i]-mean_val);
    }

    var_mean = temp_sum / (len_data);

    // cout << var_mean << endl;

    std_val = sqrt(var_mean);

    // cout << std_val << endl;

    // cout << max_val << "  " << min_val << endl;

    cout << "check point : 2" << endl;

    for (int i=0; i<len_data; i++){
        result[i] = (data_arr[i]-mean_val)/std_val;
    }

    cout << "check point : 3" << endl;

}




// private method : check_and_cut_data
/*************************************************
@ func :    입력 신호의 트리거 지점을 찾아
            데이터를 정제해서 반환
@ param :   float pointer 신호 데이터
@ return :  정제된 데이터 float pointer
@ remarks : 동적 할당된 변수 삭제 주의
@ version : trigger algorithm Version 1.0 alpha
@ date : 2021. 6. 15.
*************************************************/
float * Trig_al::check_and_cut_data(float* data_arr){

    int * indices = (int *)malloc(sizeof(int)*2);

    int gap_num = window_size-shift_size;

    size_t len_data = (_msize(data_arr))/sizeof data_arr[0];
    int for_length = len_data/(gap_num);

    float * mean_array = new float[for_length]{};

    for (int i=0;i<for_length;i++){

        float * temp = new float[window_size]{};

        for (int j=0;j<window_size;j++){
            temp[j]=data_arr[i*(gap_num)+j];
        }

        mean_array[i] = cal_mean_value(temp);
        delete[] temp;
    }

    for (int i=0;i<for_length;i++){
        if(mean_array[i]>trigger_val){
            indices[0] = i;
            break;
        }
    }

    for (int i=(for_length-1);i>-1;i--){
        if(mean_array[i]>trigger_val){
            indices[1] = i;
            break;
        }
    }

    int start_index = 0;
    int end_index = 0;

    if((indices[0]*gap_num-front_buf)>0){
        start_index = indices[0]*gap_num-front_buf;
    } else {
        start_index = 0;
    }

    if((indices[1]*gap_num+window_size)<full_size){
        end_index = indices[1]*gap_num+window_size;
    } else {
        end_index = full_size-1;
    }

    float * result = new float[end_index-start_index+1]{};

    int j = 0;

    for (int i=start_index; i<=end_index; i++){
        result[j] = data_arr[i];
        j++;
    }

    delete[] mean_array;
    delete[] indices;
    delete[] data_arr;

    return result;
}





// private method : check_and_cut_data
/*************************************************
@ func :    입력 신호의 트리거 지점을 찾아
            데이터를 정제해서 반환
            앞부분만 자르는 알고리즘
@ param :   float pointer 신호 데이터
@ return :  정제된 데이터 float pointer
@ remarks : 동적 할당된 변수 삭제 주의
@ version : trigger algorithm Version 1.0 alpha
@ date : 2021. 6. 15.
*************************************************/
float * Trig_al::check_and_cut_data_v2(float* data_arr){

    int indices = 0;

    int gap_num = window_size-shift_size;

    size_t len_data = (_msize(data_arr))/sizeof data_arr[0];
    int for_length = len_data/(gap_num);

    float * mean_array = new float[for_length]{};

    for (int i=0;i<for_length;i++){

        float * temp = new float[window_size]{};

        for (int j=0;j<window_size;j++){
            temp[j]=data_arr[i*(gap_num)+j];
        }

        mean_array[i] = cal_mean_value(temp);
        delete[] temp;
    }

    for (int i=0;i<for_length;i++){
        if(mean_array[i]>trigger_val){
            indices = i;
            break;
        }
    }

    int start_index = 0;

    if((indices*gap_num-front_buf)>0){
        start_index = indices*gap_num-front_buf;
    } else {
        start_index = 0;
    }

    float * result = new float[full_size-start_index+1]{};

    int j = 0;

    for (int i=start_index; i<=full_size; i++){
        result[j] = data_arr[i];
        j++;
    }

    delete[] mean_array;
    delete[] data_arr;

    return result;
}





// public method : preprocess_progressing
/*************************************************
@ func :    입력된 데이터를 가공해서 반환
@ param :   신호 데이터 float pointer
            결과 데이터 float pointer
            입력 신호의 크기 int
@ return :  
@ remarks : 동적 할당 메모리의 생성과 삭제 주의
            현재 노이즈 추가는 하지 않음
@ version : trigger algorithm Version 1.0 alpha
@ date : 2021. 6. 15.
@ the last update : 2021. 6. 29
*************************************************/
void Trig_al::preprocess_progressing(float * data_arr, float * result, int size_of_data){

    cout << "standardization start" << endl;

    float * temp_out_0 = new float[size_of_data]{};

    standardization_process(data_arr, temp_out_0, size_of_data);   

    cout << "standardization end" << endl;

    cout << "check and cut start" << endl;

    float * temp_out_1 = check_and_cut_data_v2(temp_out_0);

    cout << "check and cut end" << endl;

    // int len_data = (_msize(temp))/sizeof temp[0];
    
    cout << "fit and add noise start" << endl;

    float * temp_out_2 = fit_full_size(temp_out_1);

    
    // len_data = (_msize(temp_out))/sizeof temp_out[0];

    // float * temp_out_3 = add_noise(temp_out_2);

    // cout << "fit and add noise end" << endl;


    size_t len_data = (_msize(temp_out_2))/sizeof temp_out_2[0];

    memcpy(result, temp_out_2, sizeof(float)*len_data);

    cout << "end line in progressing" << endl;

    delete[] temp_out_0;
    // cout << 0 << endl;
    // cout << "end line in progressing" << endl;

}




// public method : change_record_sec
/*************************************************
@ func :    private 함수인 recording_sec 를 변환
@ param :   바꾸고자 하는 시간 int
@ return :  
@ remarks : 
@ version : trigger algorithm Version 1.0 alpha
@ date : 2021. 6. 15.
*************************************************/
void Trig_al::change_record_sec(int ch_time){
    record_sec = ch_time;
}




// public method : check_the_input_data
/*************************************************
@ func : 
@ param :
@ return : 
@ remarks : 
@ version : 
@ date : year, month, day
*************************************************/
int Trig_al::check_the_input_data(float * data_arr){



    return 0;
}








