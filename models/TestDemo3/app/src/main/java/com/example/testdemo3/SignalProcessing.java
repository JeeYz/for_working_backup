package com.example.testdemo3;

import android.util.Log;

import java.util.ArrayList;

public class SignalProcessing {
    InputTargetData targetdata = new InputTargetData();

    /**
     * 생성자
     * @param input -> 데이터
     * @param fullsize -> 최종 출력 데이터 사이즈
     */
    public SignalProcessing(ArrayList<Float> input, int fullsize){

        this.targetdata.setData(input);
        this.targetdata.setFullSize(fullsize);
    }

    /**
     * 신호 데이터 (메인 데이터)를 반환하는 코드
     * @return
     */
    public ArrayList<Float> getTargetData(){
        return this.targetdata.getData();
    }

    /**
     * 메인 데이터를 세팅해주는 코드
     * @param settarget
     */
    public void setTargetData(ArrayList<Float> settarget){
        this.targetdata.setData(settarget);
    }

    /**
     * 실제 실행부 메서드
     */
    public void runProcess(){
        /**
         * 데이터와 최종 출력 데이터의 길이
         */

        PreProcess prepro = new PreProcess();
        prepro.runPreProcess(targetdata);

        RunTriggerAlgorithm trigalg = new RunTriggerAlgorithm();
        trigalg.runTriggerAlgorithm(targetdata);

    }
}
