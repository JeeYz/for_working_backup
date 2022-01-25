package com.example.testdemo3;

import java.util.ArrayList;

/**
 * 음성 시그널 데이터의 추상 클래스
 */
abstract class VoiceSignalData {

    public ArrayList<Float> signaldata;
    public ArrayList<Float> meanList;

    public int frameSize;
    public int shiftSize;
    public int fullSizeOfResultData;

    public int voiceTriggerValue;
    public float triggerValueStd;
    public int decodingFrontSize;

    /** 메서드들 **/
    abstract ArrayList<Float> getData();
    abstract void setData(ArrayList<Float> targetdata);

    abstract ArrayList<Float> getMeanList();
    abstract void setMeanList(ArrayList<Float> targetMeanList);

    abstract int getFrameSize();
    abstract void setFrameSize(int frameSize);

    abstract int getShiftSize();
    abstract void setShiftSize(int shiftSize);

    abstract int getFullSizeOfResultData();
    abstract void setFullSizeOfResultData(int fullSizeOfResultData);

    abstract int getVoiceTriggerValue();
    abstract void setVoiceTriggerValue(int voiceTriggerValue);

    abstract float getTriggerValueStd();
    abstract void setTriggerValueStd(float triggerValueStd);

    abstract int getDecodingFrontSize();
    abstract void setDecodingFrontSize(int decodingFrontSize);


}

