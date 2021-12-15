package models.SignalProcessing;

import java.util.ArrayList;

public class InputTargetData extends VoiceSignalData {

    private int fullsize;

    /**
     * 생성자
     * @param fullsize
     */
    InputTargetData(){
    }

    /**
     * 소멸자
     */
    @Override
    protected void finalize() throws Throwable {
        // TODO Auto-generated method stub
        super.finalize();
    }

    @Override
    ArrayList getData(){
        return signaldata;
    }

    @Override
    void setData(ArrayList settarget){
        this.signaldata = settarget;

    }
    
}
