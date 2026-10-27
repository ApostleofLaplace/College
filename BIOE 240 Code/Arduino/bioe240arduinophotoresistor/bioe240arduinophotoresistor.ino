int photoPin = A0;

// LED Stuff
int LEDPin = 10;

// Stuff in Red 
int stepsize = 5;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600); // 115200
  pinMode(photoPin, INPUT);
  
  // LED Stuff
  pinMode(LEDPin, OUTPUT);


}

void loop() {
  // put your main code here, to run repeatedly:
  int measuredValue = analogRead(photoPin);
  Serial.println(measuredValue);
  
  // LED Stuff
  analogWrite(LEDPin, OUTPUT);
  delay(200);
  analogWrite(LEDPin, 0);
  delay(200);

  // Stuff in Red
  int i;
  i = 0;
  while (i <= 255) {
    if ( (i % 2) == 0 ) {
      analogWrite(LEDPin, i/3);
    }
    else {
      analogWrite(LEDPin, i);
    }
  delay(30);
  int measuredValue = analogRead(photoPin);
  Serial.println(measuredValue);
  i = i + stepsize;
  }
}
