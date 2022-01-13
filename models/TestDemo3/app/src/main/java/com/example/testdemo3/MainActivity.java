package com.example.testdemo3;

import androidx.appcompat.app.AppCompatActivity;

import android.media.AudioRecord;
import android.os.Bundle;

import java.io.PipedInputStream;
import java.io.PipedOutputStream;

public class MainActivity extends AppCompatActivity {

    private static AudioRecord audioRec;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        System.out.println("hello, world~!!");

        SendThreadDemo sendT = new SendThreadDemo(audioRec);
        ReceiveThreadDemo receiveT = new ReceiveThreadDemo(audioRec, sendT);

        Thread sendThread = new Thread(sendT);
        Thread receiveThread = new Thread(receiveT);

        receiveThread.setDaemon(true);

        sendThread.start();
        receiveThread.start();

    }
}