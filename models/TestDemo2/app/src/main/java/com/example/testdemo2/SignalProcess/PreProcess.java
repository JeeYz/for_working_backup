package com.example.testdemo2.SignalProcess;

import java.util.ArrayList;

/**
 * 입력 데이터의 전처리를 수행해주는 클래스
 */
class PreProcess implements CheckSignalData, ProcessingData{

    /**
     * 입력 데이터를 표준화 시켜주는 메서드
     */
    @Override
    public void standardizeData(VoiceSignalData inputClass) {
        // TODO Auto-generated method stub
        ArrayList<Float> targetdata = inputClass.getData();

        float sumValue = 0;
        ArrayList<Float> result = new ArrayList<>();

        for (float element : targetdata){
            // System.out.println(element);
            sumValue = sumValue + element;
        }

        float meanValue = sumValue/targetdata.size();

        float sumDeviation = 0;

        for (float element : targetdata){
            // System.out.println(element);
            float temp = (float) Math.pow((element-meanValue), 2);
            sumDeviation = sumDeviation+temp;
        }

        float deviation = (float) Math.sqrt(sumDeviation/targetdata.size());

        for (int i=0; i < targetdata.size() ; i++){
            float temp = (float) (targetdata.get(i)-meanValue)/deviation;
            result.add(i, temp);
        }

        inputClass.setData(result);
    }

    @Override
    public void normalizeData(VoiceSignalData inputClass) {
        // TODO Auto-generated method stub
    }

    @Override
    public int checkSize() {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    public int checkDataType() {
        // TODO Auto-generated method stub
        return 0;
    }

    public void checkDataBeing(ArrayList<Float> targetData){

    }

    /**
     * 생성자
     */
    public PreProcess(){
    }


    /**
     * 본 클래스의 메인 메서드
     * 전처리를 실행시켜주는 메서드
     * @param inputClass
     */
    void runPreProcess(VoiceSignalData inputClass){
        if (inputClass==null){
            System.out.println("PreProcess 작업중 입니다.");
            System.out.println("입력된 클래스는 null 입니다.");
            System.out.println("클래스 내의 데이터를 0값으로 패딩합니다.");
            inputClass = whenClassIsnull();
        }
        this.standardizeData(inputClass);
        // ((InputTargetData) inputClass).printStatusInputTargetDataClass();
    }

    /**
     * 제로 패딩 내의 값을 채워주는 메서드
     * @param targetSize
     * @return
     */
    private ArrayList<Float> fillZeroValue(int targetSize){
        ArrayList<Float> result = new ArrayList<>();

        for (int i=0; i<targetSize; i++){
            result.add(i, (float)0.0);
        }

        return result;
    }

    /**
     * 하드 코딩 메서드
     */
    @Override
    public InputTargetData whenClassIsnull() {
        // TODO Auto-generated method stub
        InputTargetData result = new InputTargetData();
        result.setFullSize(40000);

        ArrayList<Float> tempData = new ArrayList<>();
        tempData = fillZeroValue(40000);

        result.setData(tempData);

        return result;

    }

}