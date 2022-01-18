package com.example.testdemo3;

import java.util.ArrayList;

/**
 * 음성 시그널 데이터의 추상 클래스
 */
abstract class VoiceSignalData {

    int fullsize;
    ArrayList<Float> signaldata;
    ArrayList<Float> meanList;

    int framesize;
    int shiftsize;
    float triggervalue;
    int decodingFrontSize;

    /**
     *  메서드들
     */

    abstract ArrayList<Float> getData();
    abstract void setData(ArrayList<Float> targetdata);

    abstract ArrayList<Float> getMeanList();
    abstract void setMeanList(ArrayList<Float> targetMeanList);

    abstract int getFramesize();
    abstract int getShiftsize();
    abstract float getTriggerValue();

    public abstract String toString();
}

/**
 * 데이터의 형을 점검해주는 클래스
 */
interface CheckSignalData{
    void checkMaxMinValue(ArrayList<Float> inputClass);
    int checkSize(ArrayList<Float> inputClass);
    int checkDataType();
    InputTargetData whenClassIsnull();
}

/**
 * 음성 신호 데이터를 가공하는데 필요한 트리거 알고리즘
 */
interface TriggerAlgorithm{
    void runTrigger(VoiceSignalData inputClass);
    /**
     * 트리거 알고리즘을 통과한 데이터를 잘라주는 메서드
     * @return array list
     */
    void cutData(InputTargetData inputClass, int inputNum);
}

/**
 * 트리거 알고리즘에서 사용할 평균값 리스트를 반환
 */
interface GenMeanValueList{
    /**
     * 평균값 리스트를 생성해서 반환
     * @return 평균값 리스트 ArrayList
     */
    void runGenerator(VoiceSignalData inputClass);

}