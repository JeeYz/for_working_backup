package com.example.testdemo3;

import android.renderscript.ScriptGroup;
import android.util.Log;

import java.util.ArrayList;

/**
 * 음성의 입력됨을 디텍션 해주는 클래스
 */
public class RunTriggerAlgorithm implements TriggerAlgorithm, GenMeanValueList, CheckSignalData{

    /**
     * 오버라이드
     * 평균값 리스트를 생성해주는 메서드
     * @param "데이터가 입력되어 있는 클래스" VoiceSignalData Class
     */
    @Override
    public void runGenerator(VoiceSignalData inputClass) {
        // TODO Auto-generated method stub
        int frameSize = inputClass.getFrameSize();
        int shiftSize = inputClass.getShiftSize();

        ArrayList<Float> result = new ArrayList<>();
        ArrayList<Float> dataList = inputClass.getData();

        checkMaxMinValue(dataList);
        checkSize(dataList);

        for (int i=0; i<dataList.size(); i=i+shiftSize) {
            ArrayList <Float> tempList = new ArrayList<>();
            for (int j=i; j<(i+frameSize); j++){
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
            float meanValue = sum/frameSize;
            int idx = i/shiftSize;
            result.add(idx, meanValue);
//            Log.i("standardization mean value", String.valueOf(meanValue));

            if (i+frameSize>=dataList.size()){
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
            if ((float)meanValueList.get(i) >= (float)inputClass.getTriggerValueStd()){
                startIndex = i;
                Log.d("compare", "trigger vs mean value");
                Log.d("trigger value", String.valueOf(inputClass.getVoiceTriggerValue()));
                Log.d("mean value", String.valueOf(meanValueList.get(i))+" "+String.valueOf(startIndex)+" of "+String.valueOf(meanValueList.size()));

                break;
            }
        }

        return (int)startIndex*inputClass.getShiftSize();
    }

    /**
     * 시작 인덱스를 사용해서 실제 클래스내의 데이터를 잘라주는 메서드
     */
    @Override
    public void cutData(InputTargetData inputClass, int startIndex){
        ArrayList<Float> result = new ArrayList<>();
        int dataFullSize = inputClass.getFullSizeOfResultData();
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
        int addZeroSize = inputClass.getFullSizeOfResultData();
        Log.d("input class fullsize ", String.valueOf(inputClass.getFullSizeOfResultData())+" "+String.valueOf(addZeroSize));

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
     * 입력된 클래스가 null일 때 수행
     */
    @Override
    public InputTargetData whenClassIsnull() {
        // TODO Auto-generated method stub
        GlobalVariablesClass tempClass = new GlobalVariablesClass();
        InputTargetData result = new InputTargetData(tempClass);

        ArrayList<Float> tempData = new ArrayList<>();
        tempData = fillZeroValue(result.getFullSizeOfResultData());

        result.setData(tempData);

        return result;

    }

    /**
     * 현재 클래스의 실제 실행부분
     * @param inputClass
     */
    void runTriggerAlgorithm(InputTargetData inputClass){
        if (inputClass==null){
            System.out.println("PreProcess 작업중 입니다.");
            System.out.println("입력된 클래스는 null 입니다.");
            System.out.println("클래스 내의 데이터를 0값으로 패딩합니다.");
            inputClass = whenClassIsnull();
        }

        this.runGenerator(inputClass);
//        System.out.println(((InputTargetData) inputClass).toString());
//        Log.i("before data", ((InputTargetData)inputClass).toString());

        this.runTrigger(inputClass);
//        System.out.println(((InputTargetData) inputClass).toString());
//        Log.i("after data", ((InputTargetData)inputClass).toString());
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
