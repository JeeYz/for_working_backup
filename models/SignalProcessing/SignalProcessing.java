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
        this.targetdata.setData(input);
        this.fullsize = fullsize;
    }

    public void runProcess(){
        //System.out.println(this.targetdata.getData());
    }

}
