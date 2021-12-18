package models.SignalProcessing;

import java.util.ArrayList;

public class SignalProcessing {

    InputTargetData targetdata = new InputTargetData();
    int fullsize = -1;

    /**
     * 생성자
     * @param input -> 데이터
     * @param fullsize -> 최종 출력 데이터 사이즈
     */
    public SignalProcessing(ArrayList input, int fullsize){
        // isempty 를 사용한 예외처리
        // 
        this.targetdata.setData(input);
        this.fullsize = fullsize;
    }

    public ArrayList getTargetData(){
        return this.targetdata.getData();
    }

    public void setTargetData(ArrayList settarget){
        this.targetdata.setData(settarget);
    }
    
    public void runProcess(){
        /**
         * 데이터와 최종 출력 데이터의 길이
         */
        // System.out.println(this.targetdata.getData());
        System.out.println(this.fullsize);

        PreProcess prepro = new PreProcess();
        prepro.runPreProcess(this.targetdata);

        RunTriggerAlgorithm trigalg = new RunTriggerAlgorithm();
        trigalg.runTriggerAlgorithm(this.targetdata);

    }
}
