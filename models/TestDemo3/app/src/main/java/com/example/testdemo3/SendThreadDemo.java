package com.example.testdemo3;

import android.media.AudioRecord;
import android.util.Log;

import java.util.ArrayList;

/**
 * Send Thread
 * 음성 데이터를 입력 받아서 Receive Thread에 전달하는 Thread
 */
public class SendThreadDemo implements Runnable {

    private static AudioRecord audioRec;
    private int chunkSize;

    private short[] audioBytes;

    private RecordingData recordDataClass;

    /**
     * Send Thread의 생성자
     * @param audioRec AudioRecord 인스턴스 객체
     * @param mainChunkSize Chunk Size -> GlobalVariablesClass 에 저장되어 있음
     */
    public SendThreadDemo(AudioRecord audioRec, int mainChunkSize) {
        this.audioRec = audioRec;
        this.chunkSize = mainChunkSize;
        audioBytes = new short[this.chunkSize];
    }

    /**
     * Send Thread의 메인 수행 부분
     */
    @Override
    public void run() {
        Log.i("Status", "** Recording Start~!! **");
        try{

            audioRec.startRecording();
            while(true){
//                audioRec.startRecording();

                int readBufBytes = audioRec.read(audioBytes, 0, audioBytes.length);
//                Log.d("read buf size", String.valueOf(readBufBytes));
                ArrayList<Float> tempArray = ToFloatArrList(ShortArrToFloatArr(audioBytes));
//                Log.d("array size", String.valueOf(tempArray.size()));

//                this.recordDataClass.addBufferData(tempArray);

//                audioBytes = new short[this.chunkSize];
            }

        } catch (Exception e){
            e.printStackTrace();
            Log.d("exception", "녹음이 정상적으로 작동하지 않습니다...");
        }
    }

    private static ArrayList<Float> bytesToFloatArray(short[] bytesArray){
        ArrayList<Float> result = new ArrayList<>();
        for (int i=0; i<bytesArray.length; i+=2){
//            int int16 = (short)(((bytesArray[i] & 0xFF) << 8) | (bytesArray[i+1] & 0xFF));
            int int16 = Integer.valueOf(((bytesArray[i] & 0xFF) << 8) | (bytesArray[i+1] & 0xFF));
//            Log.d("byte value", Integer.toBinaryString(int16));
            result.add((float) int16);
        }
//        Log.d("convert array", String.valueOf(result.size()));
        return result;

    }

    /**
     * short 값을 float 값으로 바꿔줌
     * @param array
     * @return
     */
    private float[] ShortArrToFloatArr(short[] array) {
        float[] floatArr = new float[array.length];
        for(int i = 0; i < array.length; i++){
            int tempInt = (int)array[i];
            floatArr[i] = (float)tempInt;
        }
        return floatArr;
    }

    /**
     * float array를 ArrayList<Float>() 클래스로 바꿔 줌
     * @param array
     * @return
     */
    private ArrayList<Float> ToFloatArrList(float[] array) {
        ArrayList<Float> floatArrList = new ArrayList<>();
        for (float tmp : array) {
            floatArrList.add(tmp);
        }
        return floatArrList;
    }

}