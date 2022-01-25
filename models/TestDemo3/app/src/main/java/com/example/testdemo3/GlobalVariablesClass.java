package com.example.testdemo3;

import android.content.Context;

/**
 * 프로그램 전체에서 사용하는 변수들을 모아놓은 클래스
 */
public class GlobalVariablesClass {

    /**** Global Record Data Variables ****/
    private final int sampleRateSize = 16000;
    private final int recordingTime = 5;
    private final int fullSizeOfResultData = 40000;

    private int voiceTriggerValue;
    private float triggerValueStd;

    private final int chunkSize = 400;

    private final int frameSize = 400;
    private final int shiftSize = 200;
    private final int decodingFrontSize = 10000;

    private static final int permissionRequestCode = 10;

    private Context mainActPtr;

    /**************************************/

    /**
     * 생성자
     * 프로그램 전체에서 사용하는 변수들을 모아놓은 클래스
     */
    public GlobalVariablesClass() {

    }


    public static int getPermissionRequestCode() {
        return permissionRequestCode;
    }

    public int getSampleRateSize() {
        return this.sampleRateSize;
    }

    public int getRecordingTime() {
        return recordingTime;
    }

    public int getFullSizeOfResultData() {
        return fullSizeOfResultData;
    }

    public int getVoiceTriggerValue() {
        return voiceTriggerValue;
    }

    public void setVoiceTriggerValue(int voiceTriggerValue) {
        this.voiceTriggerValue = voiceTriggerValue;
    }

    public float getTriggerValueStd() {
        return triggerValueStd;
    }

    public void setTriggerValueStd(float triggerValueStd) {
        this.triggerValueStd = triggerValueStd;
    }

    public Context getMainActPtr() {
        return mainActPtr;
    }

    public void setMainActPtr(Context mainActPtr) {
        this.mainActPtr = mainActPtr;
    }


    public int getChunkSize() {
        return chunkSize;
    }

    public int getFrameSize() {
        return frameSize;
    }

    public int getShiftSize() {
        return shiftSize;
    }

    public int getDecodingFrontSize() {
        return decodingFrontSize;
    }

    @Override
    public String toString() {
        return "GlobalVariablesClass{" +
                "sampleRateSize=" + sampleRateSize +
                ", recordingTime=" + recordingTime +
                ", fullSizeOfResultData=" + fullSizeOfResultData +
                ", voiceTriggerValue=" + voiceTriggerValue +
                ", triggerValueStd=" + triggerValueStd +
                ", chunkSize=" + chunkSize +
                ", frameSize=" + frameSize +
                ", shiftSize=" + shiftSize +
                ", decodingFrontSize=" + decodingFrontSize +
                ", mainActPtr=" + mainActPtr +
                '}';
    }
}
