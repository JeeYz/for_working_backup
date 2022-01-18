package com.example.testdemo3;

import android.util.Log;

import java.util.ArrayList;

public class RunTriggerAlgorithm implements TriggerAlgorithm, GenMeanValueList, CheckSignalData{

    /**
     * 평균값 리스트를 생성해주는 메서드
     */
    @Override
    public void runGenerator(VoiceSignalData inputClass) {
        // TODO Auto-generated method stub
        int framesize = inputClass.getFramesize();
        int shiftsize = inputClass.getShiftsize();

        ArrayList<Float> result = new ArrayList<>();
        ArrayList<Float> dataList = inputClass.getData();

        checkMaxMinValue(dataList);
        checkSize(dataList);

        for (int i=0; i<dataList.size(); i=i+shiftsize) {
            ArrayList <Float> tempList = new ArrayList<>();
            for (int j=i; j<(i+framesize); j++){
                float tempOneData = dataList.get(j);

                if (tempOneData<0){
                    tempOneData = (-1)*tempOneData;
                    tempList.add((j-i), tempOneData);

                } else {
                    tempList.add((j-i), tempOneData);
                }
            }

            float sum = (float) 0.0f;
            for (float onedata: tempList){
                sum = sum+onedata;
            }
            float meanValue = sum/framesize;
            int idx = i/shiftsize;
            result.add(idx, meanValue);
//            Log.i("standardization mean value", String.valueOf(meanValue));

            if (i+framesize>=dataList.size()){
                break;
            }
        }
        // System.out.println(result);
        inputClass.setMeanList(result);
//        Log.d("mean list size", String.valueOf(result));

        checkMaxMinValue(result);
        checkSize(result);
//
//        for (float onedata: result){
//            Log.d("one data", String.valueOf(onedata));
//        }

    }

    /**
     * 트리거를 수행해주는 메서드
     */
    @Override
    public void runTrigger(VoiceSignalData inputClass) {
        // TODO Auto-generated method stub
        int startIndex = returnStartIndex((InputTargetData)inputClass);
        addZeroPadding((InputTargetData)inputClass);
        // System.out.println(inputClass.getData().size());

        cutData((InputTargetData)inputClass, startIndex);
    }

    /**
     * 평균값 리스트와 트리거 값을 사용해서 시작 인덱스를 출력해주는 메서드
     * @param inputClass
     * @return startIndex
     */
    private int returnStartIndex(InputTargetData inputClass){
        ArrayList<Float> meanValueList = inputClass.getMeanList();

        int startIndex = 0;
        for (int i=0; i<meanValueList.size(); i++){
            if ((float)meanValueList.get(i) >= (float)inputClass.getTriggerValue()){
                startIndex = i;
                Log.d("compare", "trigger vs mean value");
                Log.d("trigger value", String.valueOf(inputClass.getTriggerValue()));
                Log.d("mean value", String.valueOf(meanValueList.get(i))+" "+String.valueOf(startIndex)+" of "+String.valueOf(meanValueList.size()));

                break;
            }
        }

        return (int)startIndex*inputClass.getShiftsize();
    }

    /**
     * 시작 인덱스를 사용해서 실제 클래스내의 데이터를 잘라주는 메서드
     */
    @Override
    public void cutData(InputTargetData inputClass, int startIndex){
        ArrayList<Float> result = new ArrayList<>();
        int dataFullSize = inputClass.getFullSize();
        Log.d("cut status full size", String.valueOf(dataFullSize));
        int decodeFrontSize = inputClass.getDecodingFrontSize();

        int newStartIndex = startIndex+dataFullSize-decodeFrontSize;
        int endIndex = newStartIndex+dataFullSize;
        Log.d("Index Status", String.format("start : %d, end : %d, gap : %d", newStartIndex, endIndex, (endIndex-newStartIndex)));

        ArrayList<Float> tempData = inputClass.getData();

        int jNumber = 0;
        for (int i=0; i<tempData.size(); i++){
            if (newStartIndex<=i && i<endIndex){
                result.add(jNumber, (Float)tempData.get(i));
                jNumber++;
            }
        }

        // System.out.println(result.size());
        // System.out.println(result);
        inputClass.setData(result);
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
     * 데이터에 제로패딩을 붙여주는 메서드
     * @param inputClass
     */
    private void addZeroPadding(InputTargetData inputClass){
        ArrayList<Float> sigData = inputClass.getData();
        int addZeroSize = inputClass.getFullSize();
        Log.d("input class fullsize ", String.valueOf(inputClass.getFullSize())+" "+String.valueOf(addZeroSize));

        ArrayList<Float> frontZero = fillZeroValue(addZeroSize);
        ArrayList<Float> backZero = fillZeroValue(addZeroSize);

        frontZero.addAll(sigData);
        frontZero.addAll(backZero);
        // System.out.println(frontTemp.size());

        inputClass.setData(frontZero);
        checkSize(frontZero);

    }

    /**
     * 하드 코딩 메서드
     * 재로패딩 full size가 하드 코딩 되어 있음.
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

    /**
     * 현재 클래스의 실제 실행부분
     * @param inputClass
     */
    void runTriggerAlgorithm(VoiceSignalData inputClass){
        if (inputClass==null){
            System.out.println("PreProcess 작업중 입니다.");
            System.out.println("입력된 클래스는 null 입니다.");
            System.out.println("클래스 내의 데이터를 0값으로 패딩합니다.");
            inputClass = whenClassIsnull();
        }

        this.runGenerator(inputClass);
//        System.out.println(((InputTargetData) inputClass).toString());
        Log.i("before data", ((InputTargetData)inputClass).toString());

        this.runTrigger(inputClass);
//        System.out.println(((InputTargetData) inputClass).toString());
        Log.i("after data", ((InputTargetData)inputClass).toString());
    }

    @Override
    public void checkMaxMinValue(ArrayList<Float> targetData) {
        float tempMaxValue = 0.0f;
        float tempMinValue = 0.0f;

        for (float element : targetData){
            if (element > tempMaxValue){
                tempMaxValue = element;
            }
            if (element < tempMinValue){
                tempMinValue = element;
            }
        }

        Log.d("*** in trigger algorithm", "generating mean value list");
        Log.d("***min max value", "MinValue : "+String.valueOf(tempMinValue)+" , "+"MaxValue : "+String.valueOf(tempMaxValue));
    }

    @Override
    public int checkSize(ArrayList<Float> tempCheckData) {
        // TODO Auto-generated method stub
        Log.d("*** in trigger algorithm", "generating mean value list");
        Log.d("**check Size : ", String.valueOf(tempCheckData.size()));
        return 0;
    }

    @Override
    public int checkDataType() {
        // TODO Auto-generated method stub
        return 0;
    }
}
