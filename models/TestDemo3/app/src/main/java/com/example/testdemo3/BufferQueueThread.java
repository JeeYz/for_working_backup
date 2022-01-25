package com.example.testdemo3;

import android.media.AudioRecord;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;

import java.util.ArrayList;

public class BufferQueueThread implements Runnable {
    private GlobalVariablesClass globalVariables;
    private Handler dataHandler;

    private static AudioRecord audioRec;
    private short[] audioBytes;

    private RecordingData recordingDataClass;
//    private ArrayList<Float> bufferQueueData;
    private float[] bufferQueueData;

    private int sampleRateSize;
    private int recordingTime;
    private int fullsizeInputVoice;
    private int fullSizeOfResultData;
    private int voiceTriggerValue;
    private int chunkSize;


    private int toProcessingDataNum = 0;

    public BufferQueueThread(AudioRecord audioRec, GlobalVariablesClass globalVariables, RecordingData recordingDataClass) {
        this.audioRec = audioRec;
        this.globalVariables = globalVariables;

        this.chunkSize = globalVariables.getChunkSize();
        this.voiceTriggerValue = globalVariables.getVoiceTriggerValue();
        this.sampleRateSize = globalVariables.getSampleRateSize();
        this.recordingTime = globalVariables.getRecordingTime();
        this.fullsizeInputVoice = this.sampleRateSize*this.recordingTime;
        this.fullSizeOfResultData = globalVariables.getFullSizeOfResultData();

        this.audioBytes = new short[this.chunkSize];
//        this.bufferQueueData = new ArrayList<>();
        this.bufferQueueData = new float[this.fullsizeInputVoice];

        this.recordingDataClass = recordingDataClass;

    }


    private int getToProcessingDataNum() {
        return toProcessingDataNum;
    }

    private void setToProcessingDataNum(int toProcessingDataNum) {
        this.toProcessingDataNum = toProcessingDataNum;
    }


    /**
     * 사이즈 크기를 입력 받아서 Queue의 앞부분을 사이즈 크기만큼 삭제
     * FIFO 구현
     * @param deleteSize
     */
    private void deleteArrayHead(int deleteSize){
        int j = 0;
        for (int i=deleteSize; i<this.bufferQueueData.length; i++){
            this.bufferQueueData[j] = this.bufferQueueData[i];
            j++;
        }

//        for (int i=0; i<deleteSize; i++){
//            this.bufferQueueData.remove(0);
//        }
    }

    /**
     * 입력된 Chunk Size 만큼의 데이터 크기의 평균값을 구하는 메서드
     * @param inputArray
     * @return
     */
    private float calculateMeanValue(float[] inputArray){
        float sumNum = 0.0f;
        float result = 0.0f;
        float temp = 0.0f;

        for (int i=0; i<this.chunkSize; i++){
            if (inputArray[i]<0){
                temp = -1*inputArray[i];
            } else {
                temp = inputArray[i];
            }
            sumNum = sumNum+temp;
        }

        result = sumNum/(float)this.chunkSize;

        return result;
    }


    private ArrayList<Float> convertToArrayList(float[] inputArray){
        ArrayList<Float> result = new ArrayList<Float>();

        for (int i=0; i<inputArray.length; i++){
            result.add(inputArray[i]);
        }

        return result;
    }

    private void addFloatArray(float[] inputArray){
        int tempIndex = this.fullsizeInputVoice-this.chunkSize;
        int inputArrayIndex = 0;
        for (int i=tempIndex; i<(tempIndex+this.chunkSize); i++){
//            int tempInt = (int)inputArray[inputArrayIndex];
//            this.bufferQueueData[i] = (float)tempInt;
            this.bufferQueueData[i] = inputArray[inputArrayIndex];
            inputArrayIndex++;
        }
    }

    @Override
    public void run() {
        try {

            while (true) {
                if (!this.recordingDataClass.isBufferDataStatus()) {
                    int readBufBytes = audioRec.read(this.audioBytes, 0, this.audioBytes.length);
//                    Log.d("read buf size", String.valueOf(readBufBytes));
                    float[] tempArray = ShortArrToFloatArr(this.audioBytes);
                    //                Log.d("array size", String.valueOf(tempArray.size())+" "+String.valueOf(tempArray.get(0)));

                    // 사이즈가 픽스된 힙 사이즈를 초과했을 때의 처리
                    if ((this.bufferQueueData.length+this.chunkSize) >= this.fullsizeInputVoice) {
                        deleteArrayHead(this.chunkSize);
                    }
                    addFloatArray(tempArray);

//                    Log.e("buffer size", String.valueOf(this.bufferQueueData.length));

                    //                System.out.println(calculateMeanValue(bufferData));

                    /** 평균값을 출력해주는 코드 **/
                    float meanFloat = calculateMeanValue(tempArray);

//                    Log.d("Mean Value", String.valueOf(meanFloat));

                    if (meanFloat > this.voiceTriggerValue || getToProcessingDataNum() > 0) {

                        setToProcessingDataNum(getToProcessingDataNum() + 1);
//                        Log.e("index num", "$$$$$$$$$$$$$$$$$$"+String.valueOf(meanFloat));

                        if (getToProcessingDataNum() >= this.fullSizeOfResultData / this.chunkSize) {
                            //                        System.out.println(this.bufferQueueData.size());


                            ArrayList<Float> fullInputData = convertToArrayList(this.bufferQueueData);
//                            Log.e("input if", "*********************************");
                            this.recordingDataClass.setBufferQueueData(fullInputData);
                            this.recordingDataClass.setBufferDataStatus(true);

                            setToProcessingDataNum(0);
                            this.bufferQueueData = new float[this.fullsizeInputVoice];
//                            audioRec.release();

                        }
                    }
                }
            }
        } catch (Exception e){
            e.printStackTrace();
            Log.d("status", "녹음이 정상적으로 동작하지 않습니다.");
        }
    }

    /**
     * short 값을 float 값으로 바꿔줌
     * @param array
     * @return
     */
    private float[] ShortArrToFloatArr(short[] array) {
//        ArrayList<Float> floatArrList = new ArrayList<>();
        float[] tempFloatArray = new float[array.length];
        for(int i = 0; i < array.length; i++){
            int tempInt = (int)array[i];
//            floatArrList.add((float)tempInt);
            tempFloatArray[i] = (float)tempInt;
        }
        return tempFloatArray;
    }

}
