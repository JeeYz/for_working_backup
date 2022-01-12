package com.example.testdemo2.SignalProcess;

import java.util.ArrayList;

public class InputTargetData extends VoiceSignalData {

    private int fullsize;
    private ArrayList<Float> signaldata;
    private ArrayList<Float> meanList;

    private int framesize = 400;
    private int shiftsize = 200;
    private float triggervalue = (float)1.0;
    private int decodingFrontSize = 8000;

    /**
     * 생성자
     * @param fullsize
     */
    InputTargetData(){
    }

    /**
     * 최종 모델에 입력 데이터의 앞부분에 넣어줄 데이터 크기
     * @return
     */
    public int getDecodingFrontSize() {
        return this.decodingFrontSize;
    }

    /**
     * 최종 출력 데이터 사이즈 반환
     * @return fullsize 사이즈
     */
    public int getFullSize(){
        return this.fullsize;
    }

    /**
     * 풀사이즈 세팅
     * @param fullSize
     */
    public void setFullSize(int fullSize){
        this.fullsize = fullSize;
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

    /**
     * 프레임 사이즈를 반환해주는 메서드
     */
    @Override
    int getFramesize() {
        // TODO Auto-generated method stub
        return this.framesize;
    }

    /**
     * 쉬프트 사이즈를 반환해주는 메서드
     */
    @Override
    int getShiftsize() {
        // TODO Auto-generated method stub
        return this.shiftsize;
    }

    /**
     * 트리거 값을 반환해주는 메서드
     */
    @Override
    float getTriggerValue() {
        // TODO Auto-generated method stub
        return this.triggervalue;
    }

    public String toString() {
        // TODO Auto-generated method stub
        ArrayList<Float> currentData = this.getData();
        ArrayList<Float> meanValueList = this.getMeanList();

        String returnStr = "Data Size : " + currentData.size()
                + " , Data Type : " + currentData.get(0).getClass().getName()
                + " , Mean Value List Size : " + meanValueList.size()
                + "\n" + "*Parameters*  "
                + "Result Data Size : " + getFullSize()
                + " , Frame Size : " + getFramesize()
                + " , Shift Size : " + getShiftsize()
                + " , Trigger Value : " + getTriggerValue()
                + " , Front Buffer Size : " + getDecodingFrontSize()
                + "\n";


        return returnStr;
    }

}
