package com.example.testdemo3;

/**
 * 음성 데이터를 표준화 혹은 정규화 해주는 클래스
 */
interface ProcessingData{
    void standardizeData(VoiceSignalData inputClass);
    void normalizeData(VoiceSignalData inputClass);

}