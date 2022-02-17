package com.example.testdemo4;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.util.Log;

public class MainActivity extends AppCompatActivity {
    Handler handler;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        handler = new Handler(){

            @Override
            public void handleMessage(@NonNull Message msg) {
                super.handleMessage(msg);
                if(msg.what == 0){
                    Log.e("value in msg", String.valueOf(msg.what) + " " + String.valueOf(msg.arg1));
                }
            }
        };

        WorkThreadDemo thread = new WorkThreadDemo(handler);
        thread.setDaemon(true);
        thread.start();



    }
}