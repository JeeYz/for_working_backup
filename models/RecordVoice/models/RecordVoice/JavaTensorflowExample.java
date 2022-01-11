package models.RecordVoice;

import java.io.File;

import org.tensorflow.lite.*;

public class JavaTensorflowExample {

    static String filePath = "D:\\PNC_ASR_2.4_CW_model_.tflite";
    static File fileClass = new File(filePath);

    public static void main(String[] args) {

        Interpreter interpreter = new Interpreter(fileClass);

        // try (Interpreter interpreter = new Interpreter(fileClass)) {

        // } catch (Exception e) {
        //       //TODO: handle exception
        //       System.out.println(e);
        // }
    }
    
}
