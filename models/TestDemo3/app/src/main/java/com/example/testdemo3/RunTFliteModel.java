package com.example.testdemo3;

import android.content.Context;
import android.os.Handler;
import android.os.Message;
import android.util.Log;

import com.example.testdemo3.ml.PncAsr24CwModel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;

/**
 * TFLite 모델을 실행 시켜주는 클래스
 */
public class RunTFliteModel {
    private Context mainAct;
    private FloatBuffer inputBuffer;
    private float[] temp;
    private int resultLabel = -1;
    private Handler labelHandler;


    public RunTFliteModel(Context fromMain, Handler labelHandler){
        this.mainAct = fromMain;
        this.labelHandler = labelHandler;
    }


    public static int findMaxResult(float[] inputResult){
        float maxValue = 0.0f;
        int maxIndex = 0;

        for (int i=0; i<inputResult.length;i++){
            Log.d(String.valueOf(i)+" ", String.valueOf(inputResult[i]));
            if (inputResult[i]>maxValue){
                maxValue = inputResult[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

//    public void runModel(ArrayList<Float> inputData) {
    public void runModel(float[] inputData) {

        try {
            PncAsr24CwModel model = PncAsr24CwModel.newInstance(this.mainAct);

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 40000}, DataType.FLOAT32);
            inputFeature0.loadArray(inputData);

            // Runs model inference and gets result.
            PncAsr24CwModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] resultFloat = outputFeature0.getFloatArray();

            int resultIndex = findMaxResult(resultFloat);

            Log.i("*** target result", String.valueOf(resultIndex));

            Message msg = Message.obtain(this.labelHandler);
            msg.what = 0;
            msg.arg1 = resultIndex;
            this.labelHandler.sendMessage(msg);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
            e.printStackTrace();
        }
    }
}
