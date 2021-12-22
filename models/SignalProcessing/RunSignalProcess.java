package models.SignalProcessing;

import java.util.ArrayList;

/**
 * 입력 데이터의 전처리를 수행해주는 클래스
 */
class PreProcess implements CheckSignalData, ProcessingData{

    /**
     * 입력 데이터를 표준화 시켜주는 메서드
     */
    @Override
    public void standardizeData(VoiceSignalData inputClass) {
        // TODO Auto-generated method stub
        ArrayList<Float> targetdata = inputClass.getData();

        float sumValue = 0;
        ArrayList<Float> result = new ArrayList<>();

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

    public void checkDataBeing(ArrayList<Float> targetData){

    }

    /**
     * 생성자
     */
    public PreProcess(){
    }


    /**
     * 본 클래스의 메인 메서드
     * 전처리를 실행시켜주는 메서드
     * @param inputClass
     */
    void runPreProcess(VoiceSignalData inputClass){
        if (inputClass==null){
            System.out.println("PreProcess 작업중 입니다.");
            System.out.println("입력된 클래스는 null 입니다.");
            System.out.println("클래스 내의 데이터를 0값으로 패딩합니다.");
            inputClass = whenClassIsnull();
        }
        this.standardizeData(inputClass);
        // ((InputTargetData) inputClass).printStatusInputTargetDataClass();
    }

    /**
     * 제로 패딩 내의 값을 채워주는 메서드
     * @param targetSize
     * @return
     */
    private ArrayList<Float> fillZeroValue(int targetSize){
        ArrayList<Float> result = new ArrayList<>();

        for (int i=0; i<targetSize; i++){
            result.add(i, (float)0.0);
        }

        return result;
    }

    /**
     * 하드 코딩 메서드
     */
    @Override
    public InputTargetData whenClassIsnull() {
        // TODO Auto-generated method stub
        InputTargetData result = new InputTargetData();
        result.setFullSize(40000);

        ArrayList<Float> tempData = new ArrayList<>();
        tempData = fillZeroValue(40000);

        result.setData(tempData);

        return result;

    }

}


/**
 * 트리거 알고리즘을 실행시켜주는 메서드
 * 실제 데이터 처리 부분
 */
class RunTriggerAlgorithm implements TriggerAlgorithm, GenMeanValueList, CheckSignalData{

    /**
     * 평균값 리스트를 생성해주는 메서드
     */
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
                float tempOneData = dataList.get(j);

                if (tempOneData<0){
                    tempOneData = (-1)*tempOneData;
                    tempList.add((j-i), tempOneData);

                } else {
                    tempList.add((j-i), tempOneData);
                }

                if(j==(i+framesize-1)){
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
        // System.out.println(result);
        inputClass.setMeanList(result);

    }

    /**
     * 트리거를 수행해주는 메서드
     */
    @Override
    public void runTrigger(VoiceSignalData inputClass) {
        // TODO Auto-generated method stub
        int startIndex = returnStartIndex((InputTargetData)inputClass);        
        addZeroPadding((InputTargetData)inputClass);
        // System.out.println(inputClass.getData().size());

        cutData((InputTargetData)inputClass, startIndex);
    }

    /**
     * 평균값 리스트와 트리거 값을 사용해서 시작 인덱스를 출력해주는 메서드
     * @param inputClass
     * @return startIndex
     */
    private int returnStartIndex(InputTargetData inputClass){
        ArrayList<Float> meanValueList = inputClass.getMeanList();
        
        int startIndex = 0;
        for (int i=0; i<meanValueList.size(); i++){
            if ((float)meanValueList.get(i) >= inputClass.getTriggerValue()){
                startIndex = i;
                break;
            }
        }

        return startIndex;
    }

    /**
     * 시작 인덱스를 사용해서 실제 클래스내의 데이터를 잘라주는 메서드
     */
    @Override
    public void cutData(InputTargetData inputClass, int startIndex){
        ArrayList<Float> result = new ArrayList<>();
        int dataFullSize = inputClass.getFullSize();
        int endIndex = startIndex+dataFullSize;

        ArrayList<Float> tempData = inputClass.getData();

        int jNumber = 0;
        for (int i=0; i<tempData.size(); i++){
            if (startIndex<=i && i<endIndex){
                result.add(jNumber, (Float)tempData.get(i));
                jNumber++;
            }
        }

        // System.out.println(result.size());
        // System.out.println(result);
        inputClass.setData(result);
    }

    /**
     * 제로 패딩 내의 값을 채워주는 메서드
     * @param targetSize
     * @return
     */
    private ArrayList<Float> fillZeroValue(int targetSize){
        ArrayList<Float> result = new ArrayList<>();

        for (int i=0; i<targetSize; i++){
            result.add(i, (float)0.0);
        }

        return result;
    }

    /**
     * 데이터에 제로패딩을 붙여주는 메서드
     * @param inputClass
     */
    private void addZeroPadding(InputTargetData inputClass){
        ArrayList<Float> sigData = inputClass.getData();
        int addZeroSize = inputClass.getFullSize();
        ArrayList<Float> frontZero = fillZeroValue(addZeroSize);
        ArrayList<Float> backZero = fillZeroValue(addZeroSize);
        ArrayList<Float> frontTemp = frontZero;
        ArrayList<Float> backTemp = backZero;

        frontTemp.addAll(sigData);
        frontTemp.addAll(backTemp);
        // System.out.println(frontTemp.size());

        inputClass.setData(frontTemp);
    }

    /**
     * 하드 코딩 메서드
     * 재로패딩 full size가 하드 코딩 되어 있음.
     */
    @Override
    public InputTargetData whenClassIsnull() {
        // TODO Auto-generated method stub
        InputTargetData result = new InputTargetData();
        result.setFullSize(40000);

        ArrayList<Float> tempData = new ArrayList<>();
        tempData = fillZeroValue(40000);

        result.setData(tempData);

        return result;

    }

    /**
     * 현재 클래스의 실제 실행부분
     * @param inputClass
     */
    void runTriggerAlgorithm(VoiceSignalData inputClass){
        if (inputClass==null){
            System.out.println("PreProcess 작업중 입니다.");
            System.out.println("입력된 클래스는 null 입니다.");
            System.out.println("클래스 내의 데이터를 0값으로 패딩합니다.");
            inputClass = whenClassIsnull();
        }

        this.runGenerator(inputClass);
        ((InputTargetData) inputClass).printStatusInputTargetDataClass();

        this.runTrigger(inputClass);
        ((InputTargetData) inputClass).printStatusInputTargetDataClass();
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

}


