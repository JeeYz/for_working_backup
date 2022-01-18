package com.example.testdemo3;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log;

import androidx.core.app.ActivityCompat;

import java.io.PipedOutputStream;

public class SendThreadDemo implements Runnable {

    private static AudioRecord audioRec;
    private byte[] audioBytes;
    private int chunkSize;

    public SendThreadDemo(AudioRecord audioRec, int mainChunkSize) {
        this.audioRec = audioRec;
        this.chunkSize = mainChunkSize;
        audioBytes = new byte[this.chunkSize];
    }

    @Override
    public void run() {

        try{
            while(true){
                audioRec.startRecording();
            }
        } catch (Exception e){
            e.printStackTrace();
            Log.d("exception", "녹음이 정상적으로 작동하지 않습니다...");
        }
    }

}