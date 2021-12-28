package models.SignalProcessing;

import java.util.ArrayList;
import java.util.Random;

public class PracticeDemo {

    /**
     * 데모 실행을 위한 테스트 코드
     * @param args
     */
    public static void main(String[] args) {
        String x = "hello, world~!!";
        System.out.println(x);

        ArrayList<Float> tempInput = genRandomInput();

        SignalProcessing mainProcess = new SignalProcessing(tempInput, 40000);
        mainProcess.runProcess();

        /**
         * 데이터를 리턴받을 코드
         */
        mainProcess.getTargetData();
        
    }

    /**
     * 테스트를 위한 임시 랜덤값 생성 메서드
     * @return ArrayList 
     */
    private static ArrayList<Float> genRandomInput(){
        int randomsize = 64000;
        int maxvalue = 64000;
        ArrayList<Float> result = new ArrayList<>();

        Random random = new Random();

        for (int i=0; i < randomsize; i++){
            float temp = (float)random.nextInt(maxvalue);
            temp = temp-(randomsize/2);
            result.add(i, temp);
        }
        return result;
    }
}
