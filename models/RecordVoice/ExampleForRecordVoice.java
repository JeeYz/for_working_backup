package models.RecordVoice;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.TargetDataLine;

public class ExampleForRecordVoice {

    public static void main(String[] args) {
        AudioFormat format = new AudioFormat(16000.0f, 16, 1, true, true);
        System.out.println(format.toString());

        TargetDataLine microphone;

        try {
            microphone = AudioSystem.getTargetDataLine(format);
            // DataLine.Info info = new DataLine.Info(TargetDataLine.class, format);
            // microphone = (TargetDataLine) AudioSystem.getLine(info);
            microphone.open(format);

            ByteArrayOutputStream out = new ByteArrayOutputStream();
            int numBytesRead;
            int CHUNK_SIZE = 400;
            byte[] data = new byte[microphone.getBufferSize() / 10]; // getBufferSize -> 16000
            // buffer size -> 1600

            microphone.start();
            // 여기까지 설정

            int bytesRead = 0;

            long beforeTime = System.currentTimeMillis();

            try {
                while (bytesRead < 100000) { // Just so I can test if recording
                                                // my mic works...
                    numBytesRead = microphone.read(data, 0, CHUNK_SIZE);
                    // System.out.println(numBytesRead); // -> 400
                    bytesRead = bytesRead + numBytesRead;
                    out.write(data, 0, numBytesRead);
                }

            } catch (Exception e) {
                e.printStackTrace();
            }

            long afterTime = System.currentTimeMillis();
            System.out.println((afterTime-beforeTime)/1000);

            byte audioData[] = out.toByteArray();

            ArrayList<Float> temp = bytesToFloatArray(audioData);

            microphone.close();

        } catch (LineUnavailableException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }


    }

    public ExampleForRecordVoice() {
    }

    @Override
    public String toString() {
        return "ExampleForRecordVoice []";
    }

    private static ArrayList<Float> bytesToFloatArray(byte[] bytesArray){
        ArrayList<Float> result = new ArrayList<>();
        for (int i=0; i<bytesArray.length; i+=2){
            short int16 = (short)(((bytesArray[i] & 0xFF) << 8) | (bytesArray[i+1] & 0xFF));
            result.add((float) int16);
        }
        return result;

    }
    
}
