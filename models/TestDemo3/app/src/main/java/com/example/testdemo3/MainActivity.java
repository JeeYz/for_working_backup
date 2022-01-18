package com.example.testdemo3;

import static android.media.AudioRecord.getMinBufferSize;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.media.AudioRecord;
import android.os.Bundle;
import android.util.Log;

import java.io.IOException;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;

import android.media.AudioFormat;
import android.media.MediaRecorder;

public class MainActivity extends AppCompatActivity {

    private static final int RECORDER_SOURCE = MediaRecorder.AudioSource.MIC;
    private static final int RECORDER_SAMPLERATE = 16000;
    private static final int RECORDER_CHANNELS = AudioFormat.CHANNEL_IN_MONO;
    private static final int RECORDER_AUDIO_ENCODING = AudioFormat.ENCODING_PCM_16BIT;
    private static int bufferSizeInBytes = 0;
    private static int chunkSize = 800;

    private static AudioRecord audioRec;

    private RecordingData recordingDataClass = new RecordingData(this);

    private static final String[] PERMISSIONS = {
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.MANAGE_EXTERNAL_STORAGE
    };

    private static final int PERMISSIONS_REQUEST_CODE = 10;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            // TODO: Consider calling
            //    ActivityCompat#requestPermissions
            // here to request the missing permissions, and then overriding
            //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
            //                                          int[] grantResults)
            // to handle the case where the user grants the permission. See the documentation
            // for ActivityCompat#requestPermissions for more details.
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, PERMISSIONS_REQUEST_CODE);
            return;
        }

        Log.i("audio permission", String.valueOf(ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)));

        audioRec = new AudioRecord(
                RECORDER_SOURCE,
                RECORDER_SAMPLERATE,
                RECORDER_CHANNELS,
                RECORDER_AUDIO_ENCODING,
                this.chunkSize
        );

        SendThreadDemo sendT = new SendThreadDemo(audioRec, this.chunkSize);
        ReceiveThreadDemo receiveT = new ReceiveThreadDemo(audioRec, this.chunkSize, this.recordingDataClass);

        Thread sendThread = new Thread(sendT);
        Thread receiveThread = new Thread(receiveT);

        System.out.println("hello, world~!!");

        receiveThread.setDaemon(true);

        sendThread.start();
        receiveThread.start();

    }

//    private boolean hasPermission() {
//        int Result1 = ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE);
//        int Result2 = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO);
//        int Result3 = ContextCompat.checkSelfPermission(this, Manifest.permission.MANAGE_EXTERNAL_STORAGE);
//
//        Log.d("permission result", String.valueOf(Result1)
//                        + " " + String.valueOf(Result2)
//                        + " " + String.valueOf(Result3)
//        );
//
//        boolean Result = (Result1 == PackageManager.PERMISSION_GRANTED
//                && Result2 == PackageManager.PERMISSION_GRANTED
//                && Result3 == PackageManager.PERMISSION_GRANTED);
//
//        Log.d("permission result", String.valueOf(Result));
//        return Result;
//    }
//
//    private void requestPermission() {
//        ActivityCompat.requestPermissions(this, PERMISSIONS, PERMISSIONS_REQUEST_CODE);
//    }
//
//

}