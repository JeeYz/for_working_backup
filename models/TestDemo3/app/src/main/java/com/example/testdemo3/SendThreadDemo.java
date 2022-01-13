package com.example.testdemo3;

import static android.media.AudioRecord.getMinBufferSize;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log;

import androidx.core.app.ActivityCompat;

public class SendThreadDemo implements Runnable {

    private static final int RECORDER_SOURCE = MediaRecorder.AudioSource.MIC;
    private static final int RECORDER_SAMPLERATE = 16000;
    private static final int RECORDER_CHANNELS = AudioFormat.CHANNEL_IN_MONO;
    private static final int RECORDER_AUDIO_ENCODING = AudioFormat.ENCODING_PCM_16BIT;
    private static int bufferSizeInBytes = 0;
    private static boolean isRecording = true;

    private AudioRecord audioRec;


    public SendThreadDemo(AudioRecord audioRec) {
        this.bufferSizeInBytes = getMinBufferSize(
                RECORDER_SAMPLERATE,
                RECORDER_CHANNELS,
                RECORDER_AUDIO_ENCODING
        );

        Log.i("buffer size", String.valueOf(bufferSizeInBytes));
        this.audioRec = audioRec;

    }

    public int getBufferSizeInBytes() {
        return this.bufferSizeInBytes;
    }

    @Override
    public void run() {

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            // TODO: Consider calling
            //    ActivityCompat#requestPermissions
            // here to request the missing permissions, and then overriding
            //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
            //                                          int[] grantResults)
            // to handle the case where the user grants the permission. See the documentation
            // for ActivityCompat#requestPermissions for more details.
            return;
        }
        audioRec = new AudioRecord(
                RECORDER_SOURCE,
                RECORDER_SAMPLERATE,
                RECORDER_CHANNELS,
                RECORDER_AUDIO_ENCODING,
                bufferSizeInBytes
        );

        audioRec.startRecording();
        System.out.println(audioRec.getRecordingState());


        Log.i("current state", "end recording????");
        System.out.println(audioRec.getRecordingState());
        audioRec.stop();

    }

}