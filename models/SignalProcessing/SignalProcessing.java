package models.SignalProcessing;
import java.util.ArrayList;

/**
 * 신호처리 부분을 통합한 클래스
 */
public class SignalProcessing {

    InputTargetData targetdata = new InputTargetData();

    /**
     * 생성자
     * @param input -> 데이터
     * @param fullsize -> 최종 출력 데이터 사이즈
     */
    public SignalProcessing(ArrayList<Float> input, int fullsize){
        
        targetdata.setData(input);
        targetdata.setFullSize(fullsize);
    }

    public void printStatusSignalProcessingClass(){
        ArrayList<Float> currentData = targetdata.getData();
        System.out.println("\n");
        System.out.println("**About Signal Data**");
        System.out.println("Data Class Name : " + currentData.getClass());
        System.out.println("Data Size : " + currentData.size());
        System.out.println("Data Type : " + currentData.get(0).getClass().getName());
        System.out.println("\n");
    }

    /**
     * 신호 데이터 (메인 데이터)를 반환하는 코드
     * @return
     */
    public ArrayList<Float> getTargetData(){
        return targetdata.getData();
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
        printStatusSignalProcessingClass();

        RunTriggerAlgorithm trigalg = new RunTriggerAlgorithm();
        trigalg.runTriggerAlgorithm(targetdata);
        printStatusSignalProcessingClass();

    }
}
