package models.RecordVoice;

import java.io.PipedInputStream;
import java.io.PipedOutputStream;

public class ThreadProcessDemo {

    public static void main(String[] args) {

        PipedOutputStream toRecieveThread = null;
        PipedInputStream fromSendThread = null;

        // SendDataThread SendThread = null;
        // RecieveDataThread RecieveThread = null;

        try {
            toRecieveThread = new PipedOutputStream();
            fromSendThread = new PipedInputStream(toRecieveThread);

            SendDataThread SendThread = new SendDataThread(toRecieveThread);
            RecieveDataThread RecieveThread = new RecieveDataThread(fromSendThread, SendThread.getCHUNK_SIZE());
            Thread SendT  = new Thread(SendThread);
            Thread RecieveT = new Thread(RecieveThread);

            RecieveT.setDaemon(true);

            SendT.start();
            RecieveT.start();

            try {
                SendT.join();
            } catch (Exception e) {
                //TODO: handle exception
            }

            // SendT.run();
            // RecieveT.run();

        } catch (Exception e) {
            //TODO: handle exception
            System.out.println("스레드가 정상 작동하지 않습니다.");
        }
        
    }

}
