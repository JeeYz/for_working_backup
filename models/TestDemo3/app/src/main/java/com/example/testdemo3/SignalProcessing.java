package com.example.testdemo3;

import android.provider.Settings;
import android.util.Log;

import java.util.ArrayList;


/**
 * 신호처리 전반을 수행하는 클래스
 */
public class SignalProcessing {
    private static InputTargetData targetData;
    private GlobalVariablesClass globalVariables;

    private static PreProcess preProClass;
    private static RunTriggerAlgorithm trigAlgClass;

    /**
     * 생성자
     * 신호처리 전반을 수행하는 클래스
     * 전처리부터 필요한 부분을 잘라 주는 알고리즘을 실행시키는 클래스
     * @param globalVariables -> 전역 변수가 저장되어 있는 클래스
     */
    public SignalProcessing(GlobalVariablesClass globalVariables){

        this.globalVariables = globalVariables;
        this.targetData = new InputTargetData(globalVariables);
        this.preProClass = new PreProcess();
        this.trigAlgClass = new RunTriggerAlgorithm();

    }

    public void initInputDataClass(){
        this.targetData = new InputTargetData(this.globalVariables);
    }

    /**
     * 신호 데이터 (메인 데이터)를 반환하는 코드
     * @return
     */
    public ArrayList<Float> getTargetData(){
        return this.targetData.getData();
    }

    /**
     * 메인 데이터를 세팅해주는 코드
     * @param settarget
     */
    public void setTargetData(ArrayList<Float> settarget){
        this.targetData.setData(settarget);
    }

    /**
     * 실제 실행부 메서드
     */
    public void runProcess(ArrayList<Float> input){
        /**
         * 데이터와 최종 출력 데이터의 길이
         */

        this.targetData.setData(input);

        this.preProClass.runPreProcess(this.targetData);
        this.trigAlgClass.runTriggerAlgorithm(this.targetData);

    }
}
