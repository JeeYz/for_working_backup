package com.example.testdemo4;

import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.util.Log;

public class WorkThreadDemo extends Thread {
    int value = 0;
    Handler handler;

    public WorkThreadDemo(Handler handler){

        this.handler = handler;
    }

    public void run(){
        for (int i=0; i<10; i++){
            try{
                Thread.sleep(1000);
            } catch (Exception e){
                e.printStackTrace();
            }

            value += 1;

            Message msg = Message.obtain(handler);
//            Message msg = new Message();
            msg.what = 0;
            msg.arg1 = value;

            this.handler.sendMessage(msg);

            try{
                Thread.sleep(100);
            } catch (Exception e){
                e.printStackTrace();
            }

        }
    }

}
