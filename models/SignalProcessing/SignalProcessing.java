package models.SignalProcessing;

import java.util.ArrayList;

public class SignalProcessing {
    /**
     * 생성자
     * @param input -> 데이터
     * @param fullsize -> 최종 출력 데이터 사이즈
     */
    private SignalProcessing(ArrayList input, int fullsize){
        InputTargetData targetdata = new InputTargetData(fullsize);
        targetdata.setData(input);
    }

    public void runProcess(){

    }

}
