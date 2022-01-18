package com.example.testdemo3;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import com.example.testdemo3.ml.PncAsr24CwModel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;

public class RunTFliteModel {
    private Context mainAct;
    private FloatBuffer inputBuffer;
    private float[] temp;


    public RunTFliteModel(Context fromMain){
        this.mainAct = fromMain;
    }

    public static byte[] floatToByteArray(float value) {
        int intBits =  Float.floatToIntBits(value);
        return new byte[] {
                (byte) (intBits >> 24), (byte) (intBits >> 16), (byte) (intBits >> 8), (byte) (intBits) };
    }

    public static int findMaxResult(float[] inputResult){
        float maxValue = 0.0f;
        int maxIndex = 0;

        for (int i=0; i<inputResult.length;i++){
            if (inputResult[i]>maxValue){
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public void runModel(ArrayList<Float> inputData) {
//        inputBuffer = FloatBuffer.allocate(inputData.size());
//        for (float element: inputData){
//            inputBuffer.put(element);
//        }

        temp = new float[inputData.size()];

        for (int i=0; i < inputData.size(); i++){
            temp[i] = inputData.get(i);
        }

        try {
            PncAsr24CwModel model = PncAsr24CwModel.newInstance(this.mainAct);

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 40000}, DataType.FLOAT32);
            inputFeature0.loadArray(temp);

            // Runs model inference and gets result.
            PncAsr24CwModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] resultFloat = outputFeature0.getFloatArray();

            int resultIndex = findMaxResult(resultFloat);

            Log.i("*** target result", String.valueOf(resultIndex));

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
            e.printStackTrace();
        }
    }
}
