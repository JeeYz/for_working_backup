package com.example.testdemo5;

import android.content.Context;
import android.os.Handler;
import android.util.Log;

import java.util.ArrayList;

public class RunSignalProcessTfLiteModel implements Runnable {
    private RecordingData recordingDataClass;
    private SignalProcessing signalProcessClass;
    private RunTFliteModel tfLiteModel;
    private GlobalVariablesClass globalVariables;
    private Context mainAct;

    public RunSignalProcessTfLiteModel(GlobalVariablesClass globalVariables,
                                       RecordingData recordingDataClass,
                                       Context fromMainAct,
                                       Handler labelHandler) {
        this.recordingDataClass = recordingDataClass;
        this.globalVariables = globalVariables;
        this.mainAct = fromMainAct;
        this.tfLiteModel = new RunTFliteModel(this.mainAct, labelHandler, globalVariables);
        this.signalProcessClass = new SignalProcessing(this.globalVariables);
    }

    @Override
    public void run() {
        while (true){
            if (this.recordingDataClass.isBufferDataStatus()){
                ArrayList<Float> temp = this.recordingDataClass.getBufferQueueData();
                Log.d("size", String.valueOf(temp.size()));
                Log.d("value", String.valueOf(temp.get(0)));

                this.signalProcessClass.runProcess(temp);
                ArrayList<Float> tempList = this.signalProcessClass.getTargetData();

                float [] inputTemp = new float[tempList.size()];
                Log.d("size of input data", String.valueOf(tempList.size()));
                for (int i=0; i < tempList.size(); i++){
                    inputTemp[i] = tempList.get(i);
                }
//                for (int i=0; i < tempList.size(); i++){
//                    inputBuffer.array()[i] = tempList.get(i);
//                }
                Log.d("status", "Done Converting");
//                this.tfLiteModel.runModel(this.signalProcessClass.getTargetData());
                this.tfLiteModel.runModel(inputTemp, globalVariables);

//                this.tfLiteModel.Invoke(inputBuffer);

                this.signalProcessClass.initInputDataClass();
                this.recordingDataClass.initData();
            }
        }
    }

    private ArrayList<Float> ToFloatArrList(float[] array) {
        ArrayList<Float> floatArrList = new ArrayList<Float>();
        for (float tmp : array) {
            floatArrList.add(tmp);
        }
        return floatArrList;
    }

}
