package com.example.testdemo3;

/**
 * 음성 신호 데이터를 가공하는데 필요한 트리거 알고리즘
 */
interface TriggerAlgorithm{
    void runTrigger(VoiceSignalData inputClass);
    /**
     * 트리거 알고리즘을 통과한 데이터를 잘라주는 메서드
     * @return array list
     */
    void cutData(InputTargetData inputClass, int inputNum);
}