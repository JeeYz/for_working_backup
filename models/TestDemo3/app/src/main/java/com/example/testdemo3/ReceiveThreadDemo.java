package com.example.testdemo3;


import android.app.Activity;
import android.media.AudioRecord;
import android.util.Log;

import java.io.PipedInputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;

public class ReceiveThreadDemo implements Runnable{

    private ArrayList<Float> tempArray;

//    private byte[] audioBytes;
    private short[] audioBytes;
    private AudioRecord audioRec;
    private int chunkSize;

    private RecordingData recordDataClass;


    public ReceiveThreadDemo(AudioRecord mainAudioRec, int chunkSize, RecordingData recordingDataClass){
        this.audioRec = mainAudioRec;
        this.chunkSize = chunkSize;
        this.audioBytes = new short[chunkSize];
        this.recordDataClass = recordingDataClass;

//        Log.d("check instance", String.valueOf(this.recordDataClass.getFullSizeOfData()));
    }


    private void detectMaxMinValue(ArrayList<Float> inputArray){
        float tempMax = 0.0f;
        float tempMin = 0.0f;
        for (float element : inputArray){

        }
    }


    @Override
    public void run() {
//        Log.d("check instance", String.valueOf(this.recordDataClass.getFullSizeOfData()));

//        ByteBuffer tempBuffer = ByteBuffer.allocateDirect(this.chunkSize);
        while (true) {
            try {
                int readBufBytes = audioRec.read(audioBytes, 0, audioBytes.length);
//                int readBufBytes = audioRec.read(tempBuffer, this.chunkSize);
//                Log.d("read buffer bytes", String.valueOf(readBufBytes));
//                Log.d("input read size", String.valueOf(audioBytes.length));

                tempArray = bytesToFloatArray(audioBytes);
//                Log.d("arraylist size", String.valueOf(tempArray.size()));

                this.recordDataClass.addBufferData(tempArray);
//                Log.d("Posision Debug", "after recording data class add buffer");

                audioBytes = new short[this.chunkSize];

            } catch (Exception e){
                e.printStackTrace();
                Log.e("error occured", "데이터 전송과정에서 에러가 발생했습니다...");
                return;
            }
        }
    }

    private static ArrayList<Float> bytesToFloatArray(short[] bytesArray){
        ArrayList<Float> result = new ArrayList<>();
        for (int i=0; i<bytesArray.length; i+=2){
            int int16 = (short)(((bytesArray[i] & 0xFF) << 8) | (bytesArray[i+1] & 0xFF));
            result.add((float) int16);
        }
//        Log.d("convert array", String.valueOf(result.size()));
        return result;

    }
//
//    private void writeByteArrayToFile(ArrayList<Float> tempInputArray, String FileName) throws IOException {
//        String filePath = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).toString() + "/" + FileName;
//
//        File file = new File(filePath);
//        if(!file.exists()) {
//            try {
//                file.createNewFile();
//            } catch(IOException e) {
//                e.printStackTrace();
//            }
//        }
//
//        FileOutputStream targetFile = new FileOutputStream(filePath);
//
//        try {
//            ObjectOutputStream fos = new ObjectOutputStream(targetFile);
//            fos.writeObject(tempInputArray);
//            fos.flush();
//            fos.close();
//
//        } catch(FileNotFoundException e) {
//            e.printStackTrace();
//        }
//
//    }

}
