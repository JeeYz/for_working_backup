package com.example.testdemo3;

import android.media.AudioRecord;
import android.util.Log;

import java.nio.ByteBuffer;

public class ReceiveThreadDemo implements Runnable{

    private ByteBuffer audioBuffer;
    private AudioRecord audioRec;
    private int bufferSizeInBytes;
    private int recordingState;

    public ReceiveThreadDemo(AudioRecord audioRec, SendThreadDemo sendT){
        this.audioRec = audioRec;
        this.bufferSizeInBytes = sendT.getBufferSizeInBytes();
    }

    @Override
    public void run() {
//        while (true){
//            recordingState = audioRec.getRecordingState();
//            Log.i("recording state", String.valueOf(recordingState));
//        }
//        while(audioRec.getRecordingState()){
//            int readBuf= audioRec.read(audioBuffer, 0, this.bufferSizeInBytes);
//            Log.d("read Buffer", String.valueOf(readBuf));
//        }
    }
}
