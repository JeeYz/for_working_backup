package models.SignalProcessing;

import java.util.ArrayList;

public class ConvertByteToFloat {

    public static float fromByteToFloat(byte[] inputByte){
        int asInt = (inputByte[0] & 0xFF) 
            | ((inputByte[1] & 0xFF) << 8) 
            | ((inputByte[2] & 0xFF) << 16) 
            | ((inputByte[3] & 0xFF) << 24);
        return Float.intBitsToFloat(asInt);
    }    

    // public static ArrayList<Float> fromByteToFloatArray(ArrayList<Byte> inputByte){
    //     ArrayList<Float> result = new ArrayList<>();
    //     for(int i; i<inputByte.size(); i++){
    //         byte[] byteArr = inputByte.get(i);
    //         int asInt = (onedata[0] & 0xFF) 
    //             | ((onedata[1] & 0xFF) << 8) 
    //             | ((onedata[2] & 0xFF) << 16) 
    //             | ((onedata[3] & 0xFF) << 24);
    //         result.add((float) asInt);
    //     }
    //     return result;
    // }    

}
