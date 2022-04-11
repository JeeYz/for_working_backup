package com.example.testdemo5;

import static android.os.SystemClock.sleep;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    private GlobalVariablesClass mainGlobalVariables = new GlobalVariablesClass();
    private ArrayList<String> resultLabelList = mainGlobalVariables.getResultLabelList();

    @RequiresApi(api = Build.VERSION_CODES.P)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        TextView mainTextView = findViewById(R.id.mainTextView);
        ImageView micImageView = findViewById(R.id.micImageView);

        TextView mainTextView2 = findViewById(R.id.mainTextView2);
        ImageView micImageView2 = findViewById(R.id.micImageView2);

        mainGlobalVariables.setPtrMainTextView(mainTextView);
        mainGlobalVariables.setMicImageView(micImageView);

        mainGlobalVariables.setPtrMainTextView2(mainTextView2);
        mainGlobalVariables.setMicImageView2(micImageView2);

        DecoderDemoCW24 newDecoder = new DecoderDemoCW24(this, mainGlobalVariables);

        DecoderDemoCW24.speechListener voidListener = new DecoderDemoCW24.speechListener() {
            public int resultLabel;

            @Override
            public int onLabel(int resultLabel) {

                if (resultLabel == mainGlobalVariables.getStartSaveData()){
                    mainTextView.setText(mainGlobalVariables.getDecodingTextView());
                    micImageView.setVisibility(View.INVISIBLE);

                    mainTextView2.setText(mainGlobalVariables.getDecodingTextView());
                    micImageView2.setVisibility(View.INVISIBLE);
                } else if (resultLabel == 0){
                    mainTextView.setText(mainGlobalVariables.getRetryTextView());
                    micImageView.setVisibility(View.VISIBLE);

                    mainTextView2.setText(mainGlobalVariables.getRetryTextView());
                    micImageView2.setVisibility(View.VISIBLE);
                } else {
                    String resultString = new String();
                    resultString = mainGlobalVariables.getResultLabelList().get(resultLabel) + mainGlobalVariables.getResultTextView();

                    mainTextView.setText(resultString);
                    micImageView.setVisibility(View.INVISIBLE);

                    mainTextView2.setText(resultString);
                    micImageView2.setVisibility(View.INVISIBLE);

//                    sleep(2000);
//
//                    mainTextView.setText(mainGlobalVariables.getRetryTextView());
//                    micImageView.setVisibility(View.VISIBLE);
                }

                Log.e("in main activity", String.valueOf(resultLabel));

                return resultLabel;
            }

        };
        newDecoder.onSetListener(voidListener);

        newDecoder.settingSignalPrecessing(2000, 1.0f);
//        newDecoder.settingSignalPrecessing(2500, 1.1f);
//        newDecoder.settingSignalPrecessing(500, 1.0f);
        newDecoder.runDecoder();

        Log.d("main activity text", "after decoder implement");

    }
}