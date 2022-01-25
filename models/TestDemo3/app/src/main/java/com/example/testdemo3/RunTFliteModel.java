package com.example.testdemo3;

import android.content.Context;
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


    public RunTFliteModel(Context fromMain){
        this.mainAct = fromMain;
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
//        inputBuffer = FloatBuffer.allocate(inputData.size());
//        for (float element: inputData){
//            inputBuffer.put(element);
//        }

//        temp = new float[inputData.size()];
//
//        for (int i=0; i < inputData.size(); i++){
//            temp[i] = inputData.get(i);
//        }

//        String file_of_a_tensorflowlite_model = "D:\\tflite_model\\PNC_ASR_2.4_CW_model_.tflite";
//
//        try (Interpreter interpreter = new Interpreter(file_of_a_tensorflowlite_model)) {
//            interpreter.run(input, output);
//        }

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

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
            e.printStackTrace();
        }
    }
}

//import android.content.Context;
//import android.util.Log;
//
//import org.tensorflow.lite.Interpreter;
//import org.tensorflow.lite.support.common.FileUtil;
//
//import java.io.IOException;
//import java.nio.FloatBuffer;
//import java.nio.MappedByteBuffer;
//import java.util.ArrayList;
//
//public class RunTFliteModel {
//    private Interpreter interpreter = null;
//    private static final String ModelName = "D:\\tflite_model\\PNC_ASR_2.4_CW_model_.tflite";
//    public static String res;
//
//    public RunTFliteModel(Context fromMainAct) {
//        Interpreter.Options tfliteOptions = new Interpreter.Options();
//        try {
//            MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(fromMainAct, ModelName);
//            interpreter = new Interpreter(tfliteModel, tfliteOptions);
//            Log.d("Test", "[TFLite] Load Model Success");
//        } catch(IOException e) {
//            e.printStackTrace();
//            Log.e("Test", "[TFLite] Load Model Failed");
//        }
//    }
//
//    public void Invoke(FloatBuffer inputBuffer) {
//        FloatBuffer outputBuffer = FloatBuffer.allocate(32);
//        outputBuffer.rewind();
//        if(null != interpreter) {
//            interpreter.run(inputBuffer, outputBuffer);
//            MaxResultIndex(outputBuffer);
//        }
//        else {
//            Log.e("Test", "[TFLite] Interpreter is Null");
//        }
//    }
//
//    private void MaxResultIndex(FloatBuffer output) {
//        float Max = 0f;
//        int index = 0;
//        for(int i = 0; i < 32; i++) {
//            if(output.get(i) > Max) {
//                Max = output.get(i);
//                index = i;
//            }
//        }
//        res = Labeling(index) + "(" + index + ", " + Max * 100 + "%)";
//    }
//
//    private String Labeling(int index) {
//        ArrayList<String> Label = new ArrayList<String>();
//        Label.add(0, "NONE(none)");
//        Label.add(1, "CHOICE(선택)");
//        Label.add(2, "CLICK(클릭)");
//        Label.add(3, "CLOSE(닫기)");
//        Label.add(4, "HOME(홈)");
//        Label.add(5, "END(종료)");
//        Label.add(6, "DARKEN(어둡게)");
//        Label.add(7, "BRIGHTEN(밝게)");
//        Label.add(8, "VOICE_COMMANDS(음성 명령어)");
//        Label.add(9, "PICTURE(촬영)");
//        Label.add(10, "RECORD(녹화)");
//        Label.add(11, "STOP(정지)");
//        Label.add(12, "DOWN(아래로)");
//        Label.add(13, "UP(위로)");
//        Label.add(14, "NEXT(다음)");
//        Label.add(15, "PREVIOUS(이전)");
//        Label.add(16, "PLAY(재생)");
//        Label.add(17, "REWIND(되감기)");
//        Label.add(18, "FAST_FORWARD(빨리감기)");
//        Label.add(19, "INITIAL_POSITION(처음)");
//        Label.add(20, "VOLUME_DOWN(소리 작게)");
//        Label.add(21, "VOLUME_UP(소리 크게)");
//        Label.add(22, "BIG_SCREEN(화면 크게)");
//        Label.add(23, "SMALL_SCREEN(화면 작게)");
//        Label.add(24, "FULL_SCREEN(전체 화면)");
//        Label.add(25, "MOVE(이동)");
//        Label.add(26, "FREEZE(멈춤)");
//        Label.add(27, "SHOW_ALL_WINDOWS(모든 창 보기)");
//        Label.add(28, "PHONE(전화)");
//        Label.add(29, "CALL(통화)");
//        Label.add(30, "ACCEPT(수락)");
//        Label.add(31, "REJECT(거절)");
//
//        return Label.get(index);
//    }
//}
