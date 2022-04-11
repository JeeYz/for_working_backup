package com.example.testdemo5;

import java.util.ArrayList;

/**
 * RecordingData 클래스
 * 입력되는 음성 데이터 전반의 데이터 흐름을 제어함
 * 입력, 트리거, TFLite 모델 실행 등 전반적인 흐름 제어
 */
public class RecordingData {
    private static ArrayList<Float> bufferQueueData;
    private static boolean bufferDataStatus;

    public RecordingData() {
        this.bufferQueueData = new ArrayList<>();
        this.bufferDataStatus = false;
    }

    public static ArrayList<Float> getBufferQueueData() {
        return bufferQueueData;
    }

    public static void setBufferQueueData(ArrayList<Float> bufferQueueData) {
        RecordingData.bufferQueueData = bufferQueueData;
    }

    public static boolean isBufferDataStatus() {
        return bufferDataStatus;
    }

    public static void setBufferDataStatus(boolean bufferDataStatus) {
        RecordingData.bufferDataStatus = bufferDataStatus;
    }

    public void initData(){
        this.bufferQueueData = new ArrayList<>();
        this.bufferDataStatus = false;
    }
}
