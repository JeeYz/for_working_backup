package models.SignalProcessing;

import java.util.ArrayList;

class PreProcess implements CheckSignalData, ProcessingData{

    private ArrayList targetdata;


    @Override
    public ArrayList standardizeData() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public ArrayList normalizeData() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public int checkSize() {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    public int checkDataType() {
        // TODO Auto-generated method stub
        return 0;
    }

    /**
     * 생성자
     */
    private PreProcess(){
    }


    void runProcess(){

    }

}


class RunTriggerAlgorithm implements TriggerAlgorithm, GenMeanValueList{

    @Override
    public ArrayList runGenerator() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public int runTrigger() {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    public ArrayList cutData() {
        // TODO Auto-generated method stub
        return null;
    }

}


public class RunSignalProcess {
    
}
