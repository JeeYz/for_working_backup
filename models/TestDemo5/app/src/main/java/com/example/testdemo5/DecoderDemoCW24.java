package com.example.testdemo5;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.AudioRouting;
import android.media.MediaRecorder;
import android.media.MicrophoneInfo;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.provider.Settings;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.util.List;

/**
 * 디코더의 메인 클래스
 * @info Ver.2.4 CW data Version Alpha Test Demo
 *
 * * 주 의 *
 * Context를 파라미터로 받는 부분 수정
 */

/**
 * 디코더 메인 클래스
 */
public class DecoderDemoCW24 {
    private Context fromMainAct;
    private static Handler labelHandler;
    private speechListener mSpeechListener;
    private int resultLabel = -1;

    /**** Global Variables Setting ****/
    /** Record Data Variables **/
    private int sampleRateSize;
    private int recordingTime;
    private int fullSizeOfResultData;

    private int voiceTriggerValue;
    private float triggerValueStd;

    private GlobalVariablesClass globalVariables;
    /**** * * * ****/

    /**** AudioRecord Setting ****/
    private static AudioRecord audioRec;

    private static final int RECORDER_SOURCE = MediaRecorder.AudioSource.MIC;
    private static final int RECORDER_CHANNELS = AudioFormat.CHANNEL_IN_MONO;
    private static final int RECORDER_AUDIO_ENCODING = AudioFormat.ENCODING_PCM_16BIT;

    private static int RECORDER_SAMPLERATE;
    private static int chunkSize;
    /** * * * * * * * * * * * * * * * * * * * **/

    /**** Permission Variable ****/
    private static int PERMISSIONS_REQUEST_CODE;

    /**** Declaration RecordingData Class ****/
    private static RecordingData recordingDataClass;

//    private TextView mainTextView;
//    private ImageView micImageView;

    /**
     * 메인 클래스의 생성자
     * Recording 환경을 자동 설정
     * @param fromMainAct
     */
    public DecoderDemoCW24(Context fromMainAct, GlobalVariablesClass mainGlobalVar) {
        this.fromMainAct = fromMainAct;
//        this.mainTextView = mainTextView;
//        this.micImageView = micImageView;
        this.globalVariables = mainGlobalVar;
        this.settingRecordingEnvironment();
    }


    /**
     * 녹음 기본 환경을 설정해주는 메서드
     * Sample rate, Recording Time, result Output Size, Chunk size 설정
     * Permission Request Code 설정
     * 참고 -> GlobalVariablesClass 의 필드 참고
     */
    public void settingRecordingEnvironment(){
        this.sampleRateSize = this.globalVariables.getSampleRateSize();
        this.recordingTime = this.globalVariables.getRecordingTime();
        this.fullSizeOfResultData = this.globalVariables.getFullSizeOfResultData();
        this.chunkSize = this.globalVariables.getChunkSize();

        this.RECORDER_SAMPLERATE = this.sampleRateSize;
        this.PERMISSIONS_REQUEST_CODE = this.globalVariables.getPermissionRequestCode();
    }

    /**
     * 신호 처리 부분의 환경 변수를 설정해 주는 메서드
     * Main Activity의 Context를 GlobalVariablesClass에 setting해줌
     * @param voiceTriggerValue
     * @param triggerValue
     */
    public void settingSignalPrecessing(
            int voiceTriggerValue,
            float triggerValue) {

        this.voiceTriggerValue = voiceTriggerValue;
        this.triggerValueStd = triggerValue;

        this.globalVariables.setVoiceTriggerValue(this.voiceTriggerValue);
        this.globalVariables.setTriggerValueStd(this.triggerValueStd);

        this.globalVariables.setMainActPtr(this.fromMainAct);

        this.recordingDataClass = new RecordingData();

    }



    public int getResultLabel() {
        return resultLabel;
    }

    public void setResultLabel(int resultLabel) {
        this.resultLabel = resultLabel;
    }


    /**
     * 리스너
     */
    public interface speechListener {
        int onLabel(int resultLabel);
//        int getResultLabel();
//        void setResultLabel(int resultLabel);
    }

    public void onSetListener(speechListener inputInMainActInstance){
        mSpeechListener = inputInMainActInstance;
    }


    /**
     * 디코더를 실행 시켜 주는 메서드
     * Send Thread와 Receive Thread를 실행 시켜줌
     * Receive Thread는 Deamon Thread로 설정
     */
    @RequiresApi(api = Build.VERSION_CODES.P)
    public void runDecoder() {

//        Log.e("mic condition", String.valueOf(RECORDER_SOURCE));

        /**** Permission part ***/
        // start


        // Permission end

        // temporary permission
        if (ActivityCompat.checkSelfPermission(fromMainAct, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            // TODO: Consider calling
            //    ActivityCompat#requestPermissions
            // here to request the missing permissions, and then overriding
            //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
            //                                          int[] grantResults)
            // to handle the case where the user grants the permission. See the documentation
            // for ActivityCompat#requestPermissions for more details.
            ActivityCompat.requestPermissions((Activity) fromMainAct, new String[]{Manifest.permission.RECORD_AUDIO}, this.PERMISSIONS_REQUEST_CODE);
            return;
        }

        int permissionCheck = ContextCompat.checkSelfPermission(fromMainAct, Manifest.permission.RECORD_AUDIO);
        Log.e("permission status", String.valueOf(permissionCheck));

        audioRec = new AudioRecord(
                RECORDER_SOURCE,
                RECORDER_SAMPLERATE,
                RECORDER_CHANNELS,
                RECORDER_AUDIO_ENCODING,
                this.chunkSize*5
        );

        this.labelHandler = new Handler(Looper.getMainLooper()){
            @Override
            public void handleMessage(@NonNull Message msg) {
//                Log.e("debug", "$$$$$$$@@@@@@@@@@@@");
                super.handleMessage(msg);
                if (msg.arg1 != -1){
//                    Log.e("debug", "$$$$$$$");
                    setResultLabel(msg.arg1);
                    if(mSpeechListener != null){
//                        Log.e("debug", "$$$$$$%%%%%%%%%");
                        mSpeechListener.onLabel(getResultLabel());
                        Log.d("position debug", "after handler");
                    }
                }
            }
        };


        BufferQueueThread bufT = new BufferQueueThread(
                audioRec,
                this.globalVariables,
                this.recordingDataClass,
                this.labelHandler
        );
        RunSignalProcessTfLiteModel runT = new RunSignalProcessTfLiteModel(
                this.globalVariables,
                this.recordingDataClass,
                this.fromMainAct,
                this.labelHandler
        );

        Thread bufferThread = new Thread(bufT);
        Thread runModelThread = new Thread(runT);

        System.out.println("hello, world~!!");

        bufferThread.setDaemon(true);
        runModelThread.setDaemon(true);

        audioRec.startRecording();

        bufferThread.start();
        runModelThread.start();

        Log.d("text", "after thread starting...");

    }
}