package models.SignalProcessing;

import java.util.ArrayList;

/**
 * 음성 시그널 데이터의 추상 클래스
 */
abstract class VoiceSignalData {
    /**
     * filed of data
     * 변수들
     */
    // ArrayList signaldata;
    // ArrayList meanList;

    // int framesize;
    // int shiftsize;
    // float triggervalue;

    /**
     *  메서드들
     */
    abstract ArrayList getData();
    abstract void setData(ArrayList targetdata);

    abstract ArrayList getMeanList();
    abstract void setMeanList(ArrayList targetMeanList);

    abstract int getFramesize();
    abstract int getShiftsize();
    abstract float getTriggerValue();
}

/**
 * 음성 데이터를 표준화 혹은 정규화 해주는 클래스
 */
interface ProcessingData{
    void standardizeData(VoiceSignalData inputClass);
    void normalizeData(VoiceSignalData inputClass);

}

/**
 * 데이터의 형을 점검해주는 클래스
 */
interface CheckSignalData{
    int checkSize();
    int checkDataType();
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
    // void cutData(VoiceSignalData inputClass);

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



