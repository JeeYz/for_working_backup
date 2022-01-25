package com.example.testdemo3;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.media.AudioRecord;
import android.os.Bundle;
import android.util.Log;

import android.media.AudioFormat;
import android.media.MediaRecorder;

public class MainActivity extends AppCompatActivity {

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
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, 10);
            return;
        }

        DecoderDemoCW24 newDecoder = new DecoderDemoCW24(this);
        newDecoder.settingSignalPrecessing(600, 1.5f);
        newDecoder.runDecoder();

        Log.d("main activity text", "after decoder implement");

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