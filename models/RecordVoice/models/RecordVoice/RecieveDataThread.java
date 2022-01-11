package models.RecordVoice;

import java.io.IOException;
import java.io.PipedInputStream;
import java.util.ArrayList;

public class RecieveDataThread implements Runnable{
    private PipedInputStream fromSendThread;
    private int chunkSize;
    private RecordingData recordData;

    public RecieveDataThread(PipedInputStream fromSendThread, int chunkSize) {
        this.fromSendThread = fromSendThread;
        recordData = new RecordingData();
        setChunkSize(chunkSize);
    }

    public int getChunkSize() {
        return this.chunkSize;
    }

    public void setChunkSize(int chunkSize) {
        this.chunkSize = chunkSize;
    }

    @Override
    public void run() {
        byte[] fromSendT = null;
        // TODO Auto-generated method stub
        while(true){
            try {
                // fromSendT = fromSendThread.readAllBytes();
                fromSendT = fromSendThread.readNBytes(this.chunkSize);
                // System.out.print(fromSendT.length + "  ");
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
            ArrayList<Float> temp = bytesToFloatArray(fromSendT);
            // System.out.println(temp.size());
            this.recordData.addBufferData(temp);
        }
    }

    @Override
    public String toString() {
        return "RecieveDataThread []";
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
