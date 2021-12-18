package models.SignalProcessing;

import java.util.ArrayList;

class PreProcess implements CheckSignalData, ProcessingData{

    @Override
    public void standardizeData(VoiceSignalData inputClass) {
        // TODO Auto-generated method stub
        ArrayList<Integer> targetdata = inputClass.getData();

        float sumValue = 0;
        ArrayList<Float> result = new ArrayList<Float>();

        for (float element : targetdata){
            // System.out.println(element);
            sumValue = sumValue + element;
        }

        float meanValue = sumValue/targetdata.size();

        float sumDeviation = 0;

        for (float element : targetdata){
            // System.out.println(element);
            float temp = (float) Math.pow((element-meanValue), 2);
            sumDeviation = sumDeviation+temp;
        }

        float deviation = (float) Math.sqrt(sumDeviation/targetdata.size());

        for (int i=0; i < targetdata.size() ; i++){
            float temp = (float) (targetdata.get(i)-meanValue)/deviation;
            result.add(i, temp);
        }

        // System.out.println(result);
        System.out.println(result.size());
        System.out.println(result.getClass());
        System.out.println(result.getClass().getName());
        // 밑에 있는 메서드를 사용
        System.out.println(result.get(0).getClass().getName());
        inputClass.setData(result);
    }

    @Override
    public void normalizeData(VoiceSignalData inputClass) {
        // TODO Auto-generated method stub
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

    public void checkDataBeing(ArrayList targetData){

    }

    /**
     * 생성자
     */
    public PreProcess(){
    }


    void runPreProcess(VoiceSignalData inputClass){
        this.standardizeData(inputClass);
    }

}


class RunTriggerAlgorithm implements TriggerAlgorithm, GenMeanValueList{

    @Override
    public void runGenerator(VoiceSignalData inputClass) {
        // TODO Auto-generated method stub
        int framesize = inputClass.getFramesize();
        int shiftsize = inputClass.getShiftsize();
        
        ArrayList<Float> result = new ArrayList<Float>();
        ArrayList<Float> dataList = inputClass.getData();

        for (int i=0; i<dataList.size(); i=i+shiftsize) {
            ArrayList <Float> tempList = new ArrayList<Float>();
            for (int j=i; j<(i+framesize); j++){
                if(j==(i+framesize-1)){
                    tempList.add((j-i), dataList.get(j));

                    float sum = (float) 0;
                    for (float onedata: tempList){
                        sum = sum+onedata;
                    }
                    float meanValue = sum/framesize;
                    int idx = i/framesize;
                    result.add(idx, meanValue);

                } else {
                    tempList.add((j-i), dataList.get(j));

                }
            }
            if (i+framesize>=dataList.size()){
                break;
            }
        }

        inputClass.setMeanList(result);

    }

    @Override
    public void runTrigger(VoiceSignalData inputClass) {
        // TODO Auto-generated method stub
        ArrayList meanValueList = inputClass.getMeanList();
        ArrayList sigData = inputClass.getData();
        ArrayList<Float> result = new ArrayList<>();

        for (int i=0; i<meanValueList.size(); i++){

        }

    }

    private void cutData(){

    }

    private void addZeroPadding(){

    }

    void runTriggerAlgorithm(VoiceSignalData inputClass){
        this.runGenerator(inputClass);
        // System.out.println(inputClass.getMeanList().size());
        this.runTrigger(inputClass);
    }

}


