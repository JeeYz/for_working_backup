package models.SignalProcessing;

import java.util.ArrayList;

public class InputTargetData extends VoiceSignalData {

    private int fullsize;
    private ArrayList signaldata;
    private ArrayList meanList;

    private int framesize = 400;
    private int shiftsize = 200;
    private float triggervalue = (float)1.0;
    private int decodingFrontSize = 8000;

    /**
     * 생성자
     * @param fullsize
     */
    InputTargetData(){
    }

    public int getDecodingFrontSize() {
        return decodingFrontSize;
    }

    /**
     * 소멸자
     */
    @Override
    protected void finalize() throws Throwable {
        // TODO Auto-generated method stub
        super.finalize();
    }

    public int getFullSize(){
        return this.fullsize;
    }

    public void setFullSize(int fullSize){
        this.fullsize = fullSize;
    }

    @Override
    ArrayList getData() {
        // TODO Auto-generated method stub
        return signaldata;
    }

    @Override
    void setData(ArrayList targetdata) {
        // TODO Auto-generated method stub
        signaldata = targetdata;
    }

    @Override
    ArrayList getMeanList() {
        // TODO Auto-generated method stub
        return meanList;
    }

    @Override
    void setMeanList(ArrayList targetMeanList) {
        // TODO Auto-generated method stub
        meanList = targetMeanList;
        
    }

    @Override
    int getFramesize() {
        // TODO Auto-generated method stub
        return framesize;
    }

    @Override
    int getShiftsize() {
        // TODO Auto-generated method stub
        return shiftsize;
    }

    @Override
    float getTriggerValue() {
        // TODO Auto-generated method stub
        return triggervalue;
    }


}
