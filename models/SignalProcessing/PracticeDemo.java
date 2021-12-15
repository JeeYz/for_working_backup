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

        int temp = random.nextInt(maxvalue);
        System.out.println(temp);

        SignalProcessing mainprocess = new SignalProcessing();
        
    }
    
}
