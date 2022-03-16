package com.example.testdemo3;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.media.AudioRecord;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;

import android.media.AudioFormat;
import android.media.MediaRecorder;

public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        DecoderDemoCW24 newDecoder = new DecoderDemoCW24(this);

        DecoderDemoCW24.speechListener voidListener = new DecoderDemoCW24.speechListener() {
            public int resultLabel;

            @Override
            public int onLabel(int resultLabel) {
                Log.e("in main activity", String.valueOf(resultLabel));
                return resultLabel;
            }

//            @Override
//            public int getResultLabel() {
//                return resultLabel;
//            }
//
//            @Override
//            public void setResultLabel(int resultLabel) {
//                this.resultLabel = resultLabel;
//            }
        };
        newDecoder.onSetListener(voidListener);

        newDecoder.settingSignalPrecessing(5000, 1.5f);
        newDecoder.runDecoder();

        Log.d("main activity text", "after decoder implement");

    }

}