package models.SignalProcessing;

import java.util.ArrayList;
import java.util.Random;

import javax.print.event.PrintJobListener;

public class PracticeDemo {

    public static void main(String[] args) {
        String x = "hello, world~!!";
        System.out.println(x);

        int randomsize = 64000;
        int maxvalue = 64000;

        ArrayList tempInput = new ArrayList<Integer>();
        Random random = new Random();

        for (int i=0; i < randomsize; i++){
            int temp = random.nextInt(maxvalue);
            temp = temp-(randomsize/2);
            tempInput.add(i, temp);
        }

        //System.out.println(tempInput);

        SignalProcessing mainprocess = new SignalProcessing(tempInput, 40000);
        mainprocess.runProcess();
        
    }
    
}
