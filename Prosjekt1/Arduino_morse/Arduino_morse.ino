const int ledGreen = 8;           // Indicates break
const int ledYellow = 7;           // Indicates dot
const int ledRed = 6;             // Indicates dash
const int buttonPin = 2;          // button

// Variables will change:
int buttonState;             // the current reading from the input pin
int lastButtonState = HIGH;   // the previous reading from the input pin

int startSym = 0;
int endSym = 0;
int pauseTime = 0;
int message = 0;
int pressedTime = 0;

int T = 150;

//-----------------SETUP---------------------------------------

void setup(){
  pinMode( ledGreen, OUTPUT );    // Pin 8 is an output
  pinMode( ledYellow, OUTPUT );    // Pin 7 is an output
  pinMode( ledRed, OUTPUT );    // Pin 6 is an output
  pinMode( buttonPin, INPUT);   // Takes in values from button
  Serial.begin(9600);            // Important for port output

  endSym = millis();
  
  digitalWrite(ledGreen, HIGH); // set initial LED state
  digitalWrite(ledYellow, LOW); // set initial LED state
  digitalWrite(ledRed, LOW); // set initial LED state
}


//--------------------- morse--------------------------

void morse(){
  if (buttonState == LOW){

    startSym = millis();
    pauseTime = startSym - endSym;

    
    //digitalWrite(ledGreen, HIGH);
    //digitalWrite(ledYellow, LOW);
    //digitalWrite(ledRed, LOW);
    
    //Short pause
    //if (pauseTime <= T) {
    //  message = 3;
    //}
    //Medium pause
      if (pauseTime > 2*T && pauseTime <= 3*T) {
      message = 3;
      }
    //Long pause
    if (pauseTime > 3*T) {
      message = 4;
    }
    
  }

  else{
    digitalWrite(ledGreen, LOW);

    endSym = millis();
    pressedTime = endSym - startSym;

    //Short press - dot
    if(pressedTime<=T){
      digitalWrite(ledGreen, LOW);
      digitalWrite(ledYellow, HIGH);
      digitalWrite(ledRed, LOW);

      message = 1;
    };

    //Long press - dash
    if(pressedTime>T){
      digitalWrite(ledGreen, LOW);
      digitalWrite(ledYellow, LOW);
      digitalWrite(ledRed, HIGH);

      message = 2;
    }

    delay(20);

    digitalWrite(ledGreen, HIGH);
    digitalWrite(ledYellow, LOW);
    digitalWrite(ledRed, LOW);
    
  }

  Serial.print(message);
}

//--------- Loop-----------------------------------------------
void loop(){
 buttonState = digitalRead(buttonPin);

  if (buttonState != lastButtonState) {
    morse();
  }
  
  delay(20);    //Debounce

  lastButtonState = buttonState;
  
}
