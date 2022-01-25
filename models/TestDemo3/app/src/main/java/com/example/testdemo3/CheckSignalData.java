package com.example.testdemo3;

import java.util.ArrayList;

/**
 * 데이터의 형을 점검해주는 클래스
 */
interface CheckSignalData{
    void checkMaxMinValue(ArrayList<Float> inputClass);
    int checkSize(ArrayList<Float> inputClass);
    int checkDataType();
    InputTargetData whenClassIsnull();
}
