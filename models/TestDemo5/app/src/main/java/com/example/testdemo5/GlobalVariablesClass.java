package com.example.testdemo5;
import android.content.Context;
import android.widget.ImageView;
import android.widget.TextView;

import java.util.ArrayList;

/**
 * 프로그램 전체에서 사용하는 변수들을 모아놓은 클래스
 */
public class GlobalVariablesClass {

    /**** Global Record Data Variables ****/
    private final int sampleRateSize = 16000;
    private final int recordingTime = 5;
    private final int fullSizeOfResultData = 40000;

    private int voiceTriggerValue;
    private float triggerValueStd;

    private final int chunkSize = 400;

    private final int frameSize = 400;
    private final int shiftSize = 200;
//    private final int decodingFrontSize = 6700;
    private final int decodingFrontSize = 7700;

    public float getThresholdPrediction() {
        return thresholdPrediction;
    }

    private static final float thresholdPrediction = 0.90f;

    private static final int permissionRequestCode = 10;

    private Context mainActPtr;

    private static final ArrayList<String> resultLabelList = new ArrayList<>();

    private static final String mainTextView = "명령어를 입력하세요.";
    private static final String retryTextView = "명령어를 다시 입력하세요.";
    private static final String decodingTextView = "디코딩중...";
    private static final String resultTextView = " 이(가) 입력되었습니다.";

    private static TextView ptrMainTextView;
    private static ImageView micImageView;

    public static TextView getPtrMainTextView2() {
        return ptrMainTextView2;
    }

    public static void setPtrMainTextView2(TextView ptrMainTextView2) {
        GlobalVariablesClass.ptrMainTextView2 = ptrMainTextView2;
    }

    public static ImageView getMicImageView2() {
        return micImageView2;
    }

    public static void setMicImageView2(ImageView micImageView2) {
        GlobalVariablesClass.micImageView2 = micImageView2;
    }

    private static TextView ptrMainTextView2;
    private static ImageView micImageView2;


    /**************************************/
    private final int startSaveData = 111;

    public int getStartSaveData() {
        return startSaveData;
    }


    /**************************************/

    /**
     * 생성자
     * 프로그램 전체에서 사용하는 변수들을 모아놓은 클래스
     */

    public static TextView getPtrMainTextView() {
        return ptrMainTextView;
    }

    public static void setPtrMainTextView(TextView ptrMainTextView) {
        GlobalVariablesClass.ptrMainTextView = ptrMainTextView;
    }

    public static ImageView getMicImageView() {
        return micImageView;
    }

    public static void setMicImageView(ImageView micImageView) {
        GlobalVariablesClass.micImageView = micImageView;
    }

    public static String getMainTextView() {
        return mainTextView;
    }

    public static String getRetryTextView() {
        return retryTextView;
    }

    public static String getDecodingTextView() {
        return decodingTextView;
    }

    public static String getResultTextView() {
        return resultTextView;
    }

    public GlobalVariablesClass() {
        setLabelList();
    }

    public static ArrayList<String> getResultLabelList() {
        return resultLabelList;
    }

    public static int getPermissionRequestCode() {
        return permissionRequestCode;
    }

    public int getSampleRateSize() {
        return this.sampleRateSize;
    }

    public int getRecordingTime() {
        return recordingTime;
    }

    public int getFullSizeOfResultData() {
        return fullSizeOfResultData;
    }

    public int getVoiceTriggerValue() {
        return voiceTriggerValue;
    }

    public void setVoiceTriggerValue(int voiceTriggerValue) {
        this.voiceTriggerValue = voiceTriggerValue;
    }

    public float getTriggerValueStd() {
        return triggerValueStd;
    }

    public void setTriggerValueStd(float triggerValueStd) {
        this.triggerValueStd = triggerValueStd;
    }

    public Context getMainActPtr() {
        return mainActPtr;
    }

    public void setMainActPtr(Context mainActPtr) {
        this.mainActPtr = mainActPtr;
    }


    public int getChunkSize() {
        return chunkSize;
    }

    public int getFrameSize() {
        return frameSize;
    }

    public int getShiftSize() {
        return shiftSize;
    }

    public int getDecodingFrontSize() {
        return decodingFrontSize;
    }

    private void setLabelList(){

        resultLabelList.add("None"); // 0
        resultLabelList.add("선택"); // 1
        resultLabelList.add("클릭"); // 2
        resultLabelList.add("닫기"); // 3
        resultLabelList.add("홈"); // 4
        resultLabelList.add("종료"); // 5
        resultLabelList.add("어둡게"); // 6
        resultLabelList.add("밝게"); // 7
        resultLabelList.add("음성 명령어"); // 8
        resultLabelList.add("촬영"); // 9
        resultLabelList.add("녹화"); // 10
        resultLabelList.add("정지"); // 11
        resultLabelList.add("아래로"); // 12
        resultLabelList.add("위로"); // 13
        resultLabelList.add("다음"); // 14
        resultLabelList.add("이전"); // 15
        resultLabelList.add("재생"); // 16
        resultLabelList.add("되감기"); // 17
        resultLabelList.add("빨리감기"); // 18
        resultLabelList.add("처음"); // 19
        resultLabelList.add("소리 작게"); // 20
        resultLabelList.add("소리 크게"); // 21
        resultLabelList.add("화면 크게"); // 22
        resultLabelList.add("화면 작게"); // 23
        resultLabelList.add("전체 화면"); // 24
        resultLabelList.add("이동"); // 25
        resultLabelList.add("멈춤"); // 26
        resultLabelList.add("모든 창 보기"); // 27
        resultLabelList.add("전화"); // 28
        resultLabelList.add("통화"); // 29
        resultLabelList.add("수락"); // 30
        resultLabelList.add("거절"); // 31

        return;
    }

    @Override
    public String toString() {
        return "GlobalVariablesClass{" +
                "sampleRateSize=" + sampleRateSize +
                ", recordingTime=" + recordingTime +
                ", fullSizeOfResultData=" + fullSizeOfResultData +
                ", voiceTriggerValue=" + voiceTriggerValue +
                ", triggerValueStd=" + triggerValueStd +
                ", chunkSize=" + chunkSize +
                ", frameSize=" + frameSize +
                ", shiftSize=" + shiftSize +
                ", decodingFrontSize=" + decodingFrontSize +
                ", mainActPtr=" + mainActPtr +
                '}';
    }
}
