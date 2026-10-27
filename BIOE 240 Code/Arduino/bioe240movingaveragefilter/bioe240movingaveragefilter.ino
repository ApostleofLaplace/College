#define IN_PIN A0
#define WINDOW_SIZE 50

int INDEX = 0;
int VALUE = 0;
int SUM = 0;
int READINGS[WINDOWSIZE];
int AVERAGED = 0;

void setup() {
  
  pinMode(IN_PIN, INPUT);
  
  Serial.begin(115200); // consider baud of 9600

}

void loop() {

  SUM = SUM - READINGS[INDEX];

  VALUE = analogRead(IN_PIN);

  READINGS[INDEX] = VALUE;

  SUM = SUM + VALUE;

  INDEX = ( INDEX + 1 ) % WINDOW_SIZE;

  AVERAGED = SUM / WINDOW_SIZE;

  Serial.print(VALUE);
  Serial.print(", ");
  Serial.print(AVERAGED);

  delay(25);
}
