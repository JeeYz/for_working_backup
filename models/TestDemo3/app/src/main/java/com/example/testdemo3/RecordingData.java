package com.example.testdemo3;

import android.content.Context;
import android.util.Log;

import java.util.ArrayList;

public class RecordingData {
    private SignalProcessing signalProcess;

    private ArrayList<Float> recordData;
    private ArrayList<Float> bufferQueueData;

    private int sampleRateSize = 16000;
    private int recordingTime = 5;
    private int fullsizeInputVoice = sampleRateSize*recordingTime;
    private int fullSizeOfData = 40000;
    private int voiceTriggerValue = 7000;

    private RunTFliteModel tfLiteModel;

    private int toProcessingDataNum = 0;
    private long beforeTime;

    public RecordingData(Context fromMainAct) {
        setBufferQueueData(new ArrayList<>());
        setRecordData(new ArrayList<>());
        this.tfLiteModel = new RunTFliteModel(fromMainAct);
    }

    public SignalProcessing getSignalProcess() {
        return signalProcess;
    }

    public void setSignalProcess(SignalProcessing signalProcess) {
        this.signalProcess = signalProcess;
    }

    public ArrayList<Float> getBufferQueueData() {
        return bufferQueueData;
    }

    public void setBufferQueueData(ArrayList<Float> bufferQueueData) {
        this.bufferQueueData = bufferQueueData;
    }

    public ArrayList<Float> getRecordData() {
        return this.recordData;
    }

    public void setRecordData(ArrayList<Float> recordData) {
        this.recordData = recordData;
    }

    @Override
    public String toString() {
        return "RecordingData []";
    }

    public int getToProcessingDataNum() {
        return toProcessingDataNum;
    }

    public void setToProcessingDataNum(int toProcessingDataNum) {
        this.toProcessingDataNum = toProcessingDataNum;
    }

    public int getFullSizeOfData() {
        return fullSizeOfData;
    }

    public void setFullSizeOfData(int fullSizeOfData) {
        this.fullSizeOfData = fullSizeOfData;
    }

    public void bufferSize(){
        System.out.println("Buffer Size : " + bufferQueueData.size());
    }

    public void dataSize(){
        System.out.println("Data Size : " + recordData.size());
    }

    private void deleteArrayHead(int deleteSize){
        for (int i=0; i<deleteSize; i++){
            this.bufferQueueData.remove(0);
        }
    }

    private float calculateMeanValue(ArrayList<Float> inputBuffer){
        float sumNum = 0.0f;
        float result = 0.0f;
        int bufferSize = inputBuffer.size();

        for (float element: inputBuffer){
            if (element < 0.0f){
                element = -1*element;
            }
            sumNum = sumNum + element;
        }

        result = sumNum/(float)bufferSize;

        return result;
    }

    /**
     *
     * @param bufferData
     */
    public void addBufferData(ArrayList<Float> bufferData){
        try {
            int tempSize = bufferData.size();
            this.bufferQueueData.addAll(bufferData);

            // 사이즈가 픽스된 힙 사이즈를 초과했을 때의 처리
            if (getBufferQueueData().size() > fullsizeInputVoice){
                deleteArrayHead(tempSize);
            }

//            System.out.println(calculateMeanValue(bufferData));

//            Log.d("Mean Value", String.valueOf(calculateMeanValue(bufferData)));

            if (calculateMeanValue(bufferData) > voiceTriggerValue || getToProcessingDataNum() > 0){

//                Log.d("Status", "Enter Trigger...");

                setToProcessingDataNum(getToProcessingDataNum()+1);

                if (getToProcessingDataNum() == getFullSizeOfData()/tempSize){
                    System.out.println(this.bufferQueueData.size());
                    beforeTime = System.currentTimeMillis();

                    signalProcess = new SignalProcessing(this.bufferQueueData, getFullSizeOfData());
                    signalProcess.runProcess();

                    recordData = signalProcess.getTargetData();

                    Log.d("recordData size", String.valueOf(recordData.size()));
                    Log.d("debug position", "before model");
                    // 모델이 동작하는 부분
                    tfLiteModel.runModel(recordData);
                    Log.d("debug position", "after model");

                    setToProcessingDataNum(0);

                    long afterTime = System.currentTimeMillis();
                    System.out.println("Impliment Time : " + (afterTime-beforeTime));
                }
            }

        } catch (Exception e) {
            //TODO: handle exception
            System.out.println("큐 데이터 입력에 문제가 발생했습니다.");
            System.out.println(e);
        }
    }
}
