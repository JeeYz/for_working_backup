package com.example.testdemo3;

import java.util.ArrayList;

/**
 * VoiceSignalData 클래스를 상속받은 클래스
 */
public class InputTargetData extends VoiceSignalData {

    private GlobalVariablesClass globalVariables;

    private ArrayList<Float> signaldata;
    private ArrayList<Float> meanList;

    private int frameSize;
    private int shiftSize;
    private int fullSizeOfResultData;

    private int voiceTriggerValue;
    private float triggerValueStd;
    private int decodingFrontSize;

    /**
     * 생성자
     * @param
     */
    InputTargetData(GlobalVariablesClass globalVariables){
        this.frameSize = globalVariables.getFrameSize();
        this.shiftSize = globalVariables.getShiftSize();
        this.fullSizeOfResultData = globalVariables.getFullSizeOfResultData();

        this.voiceTriggerValue = globalVariables.getVoiceTriggerValue();
        this.triggerValueStd = globalVariables.getTriggerValueStd();
        this.decodingFrontSize = globalVariables.getDecodingFrontSize();
    }

    @Override
    public int getFrameSize() {
        return frameSize;
    }
    @Override
    public void setFrameSize(int frameSize) {
        this.frameSize = frameSize;
    }

    @Override
    public int getShiftSize() {
        return shiftSize;
    }
    @Override
    public void setShiftSize(int shiftSize) {
        this.shiftSize = shiftSize;
    }

    @Override
    public int getFullSizeOfResultData() {
        return fullSizeOfResultData;
    }
    @Override
    public void setFullSizeOfResultData(int fullSizeOfResultData) {
        this.fullSizeOfResultData = fullSizeOfResultData;
    }

    @Override
    public int getVoiceTriggerValue() {
        return this.voiceTriggerValue;
    }
    @Override
    public void setVoiceTriggerValue(int voiceTriggerValue) {
        this.voiceTriggerValue = voiceTriggerValue;
    }
    @Override
    public float getTriggerValueStd() {
        return this.triggerValueStd;
    }
    @Override
    public void setTriggerValueStd(float triggerValueStd) {
        this.triggerValueStd = triggerValueStd;
    }

    @Override
    int getDecodingFrontSize() {
        return this.decodingFrontSize;
    }
    @Override
    void setDecodingFrontSize(int decodingFrontSize) {
        this.decodingFrontSize = decodingFrontSize;
    }


    /**
     * 메인 데이터를 반환해주는 메서드
     */
    @Override
    ArrayList<Float> getData() {
        // TODO Auto-generated method stub
        return this.signaldata;
    }

    /**
     * 메인 데이터를 세팅해주는 메서드
     */
    @Override
    void setData(ArrayList<Float> targetdata) {
        // TODO Auto-generated method stub
        this.signaldata = targetdata;
    }

    /**
     * 평균값 리스트를 반환해주는 메서드
     */
    @Override
    ArrayList<Float> getMeanList() {
        // TODO Auto-generated method stub
        return this.meanList;
    }

    /**
     * 평균값 리스트를 세팅해주는 메서드
     */
    @Override
    void setMeanList(ArrayList<Float> targetMeanList) {
        // TODO Auto-generated method stub
        this.meanList = targetMeanList;

    }

    @Override
    public String toString() {
        return "InputTargetData{" +
                "globalVariables=" + globalVariables +
                ", signaldata=" + signaldata +
                ", meanList=" + meanList +
                '}';
    }
}
