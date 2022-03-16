package com.example.baseapp00;

import android.graphics.Color;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
    Button button1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


//        button1 = findViewById(R.id.button1);
//        button1.setOnClickListener(new View.OnClickListener(){
//            @Override
//            public void onClick(View v){
//                Toast.makeText(getApplicationContext(), "버튼을 눌렀어요.", Toast.LENGTH_SHORT).show();
//            }
//        }
//        );

        TextView tv1, tv2, tv3;
        tv1 = findViewById(R.id.textView1);
        tv2 = findViewById(R.id.textView2);
        tv3 = findViewById(R.id.textView3);

        tv1.setText("안녕하세요..?");
        tv1.setTextColor(Color.RED);
    }
}




