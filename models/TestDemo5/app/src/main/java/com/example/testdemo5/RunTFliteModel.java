package com.example.testdemo5;

import android.content.Context;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.View;

import com.example.testdemo5.ml.PncAsr24CwModel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.FloatBuffer;

/**
 * TFLite 모델을 실행 시켜주는 클래스
 */
public class RunTFliteModel {
    private Context mainAct;
    private FloatBuffer inputBuffer;
    private float[] temp;
    private int resultLabel = -1;
    private Handler labelHandler;
    private static float thresholdPredition;


    public RunTFliteModel(Context fromMain, Handler labelHandler, GlobalVariablesClass globalVariables){
        this.mainAct = fromMain;
        this.labelHandler = labelHandler;
        this.thresholdPredition = globalVariables.getThresholdPrediction();
    }


    public int findMaxResult(float[] inputResult){
        float maxValue = 0.0f;
        int maxIndex = 0;

        for (int i=0; i<inputResult.length;i++){
            Log.d(String.valueOf(i)+" ", String.valueOf(inputResult[i]));
            if (inputResult[i]>maxValue){
                maxValue = inputResult[i];
                maxIndex = i;
            }
        }

        if (maxIndex == 7){
            if (maxValue < 0.99999f){
                Log.e("position check : ", "!!!****");
                maxValue = 0.0f;
            }
        }

        if (maxIndex == 3 || maxIndex == 11 || maxIndex == 28){
            if (maxValue > 0.8f){
                Log.e("position check : ", "!!!!!");
                maxValue = 1.0f;
            }
        }

        if (maxValue < this.thresholdPredition){
            Log.e("prediction status", String.valueOf(maxValue));
            maxIndex = 0;
        }

        return maxIndex;
    }


    public float findMaxRate(float[] inputResult){
        float maxValue = 0.0f;
        int maxIndex = 0;

        for (int i=0; i<inputResult.length;i++){
//            Log.d(String.valueOf(i)+" ", String.valueOf(inputResult[i]));
            if (inputResult[i]>maxValue){
                maxValue = inputResult[i];
                maxIndex = i;
            }
        }

        if (maxValue < this.thresholdPredition){
            Log.e("prediction status", String.valueOf(maxValue));
            maxIndex = 0;
        }

        return maxValue;
    }


    //    public void runModel(ArrayList<Float> inputData) {
    public void runModel(float[] inputData, GlobalVariablesClass globalVariables) {

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
            float resultValue = findMaxRate(resultFloat);

            Log.i("*** target prediction rate", String.valueOf(resultValue));
            Log.i("*** target result", String.valueOf(resultIndex));

//            if (resultIndex != 0){
//                globalVariables.getPtrMainTextView().setText(
//                        globalVariables.getResultLabelList().get(resultIndex)+globalVariables.getResultTextView()
//                );
//                globalVariables.getMicImageView().setVisibility(View.INVISIBLE);
//            }

            Message msg = Message.obtain(this.labelHandler);
            msg.what = 0;
            msg.arg1 = resultIndex;
            this.labelHandler.sendMessage(msg);

//            Thread.sleep(2000);
//
//            Message msgInit = Message.obtain(this.labelHandler);
//            msg.what = 0;
//            msg.arg1 = 0;
//            this.labelHandler.sendMessage(msgInit);

            // Releases model resources if no longer used.
            model.close();
//        } catch (IOException | InterruptedException e) {
        } catch (IOException e) {
            // TODO Handle the exception
            e.printStackTrace();
        }
    }
}
