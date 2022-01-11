package models.RecordVoice;

import java.io.ByteArrayOutputStream;
import java.io.PipedOutputStream;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.TargetDataLine;

public class SendDataThread implements Runnable{
    private PipedOutputStream toRecieveThread;
    private int CHUNK_SIZE = 800;

    public SendDataThread(PipedOutputStream toRecieveThread) {
        this.toRecieveThread = toRecieveThread;
    }

    public int getCHUNK_SIZE() {
        return CHUNK_SIZE;
    }

    public void setCHUNK_SIZE(int cHUNK_SIZE) {
        this.CHUNK_SIZE = cHUNK_SIZE;
    }

    @Override
    public String toString() {
        return "SendDataThread []";
    }

    @Override
    public void run() {
        // TODO Auto-generated method s
        AudioFormat format = new AudioFormat(16000.0f, 16, 1, true, true);
        System.out.println(format.toString());

        TargetDataLine microphone;
        
        try {
            microphone = AudioSystem.getTargetDataLine(format);
            // DataLine.Info info = new DataLine.Info(TargetDataLine.class, format);
            // microphone = (TargetDataLine) AudioSystem.getLine(info);
            microphone.open(format);

            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] data = new byte[microphone.getBufferSize() / 10]; // getBufferSize -> 16000
            // buffer size -> 1600

            microphone.start();
            // 여기까지 설정

            try {
                // while (bytesRead < 100000) { // Just so I can test if recording
                //                                 // my mic works...
                while (true) { // Just so I can test if recording
                                                // my mic works...
                    int numBytesRead = microphone.read(data, 0, CHUNK_SIZE);
                    out.write(data, 0, numBytesRead);
                    byte[] audioData = out.toByteArray();

                    // System.out.println("**" + audioData.length);

                    toRecieveThread.write(audioData);
                    // System.out.println("hello");

                    out.reset();
                    // System.out.println("world");
                }

            }catch (Exception e) {
                e.printStackTrace();
            }

            microphone.close();

        }catch (java.lang.IllegalArgumentException argException){ 
            System.out.println("현재 마이크가 연결되어 있지 않았을 수 있습니다.");
            System.out.println(argException);
            return;
        } catch (LineUnavailableException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}
