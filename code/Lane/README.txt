TO RUN;
    You will need:
        - A pytorch file for lane detection
        - yolov5 in the same directory with detect.py since we use their LLM to train lane detection
        - reading.py -> This contains all the logic 
        - main.py -> for running locally and testing specifically lane detection
        - adaption.py -> an adaption of Justin's code to my code, For running on the CAV
    
    COMMANDS: 

    python main.py

    sudo python3.8 adaption.py
     
     
TO CANCEL RUN;
    Keyboard interrupts such as Ctrl + C, Ctrl + Z in terminal 

IMPORTANT:
At the moment certain conditions are hardcoded. These include:
-Display Image (BOTH)
-Model Name (BOTH, Currently using pytorch)
-Video Path (BOTH)
-What frames the Center value is read (selfDriveAdapt())
-fps (selfDriveAdapt())

And you will need to manually change it in the code before running,
if runnning just for video (no CAV); reading.py -> writeToCSV()
if running on CAV with GPIO and Motor; adaption.py -> selfDriveAdapt()

Last Note: At the moment writeToCSV() and selfDriveAdapt() methods are fundamentally the same,
so any changes made to one function will need to be made to the other. 
The reason why they are separate is because all actions are currently being determined off 
the while loop, so therefore for now, selfDriveAdapt() exists specifically for running on the CAV.
With selfDriveAdapt() checking and connecting to the serial ports and sending data to the GPIO.

*************************************************************************
*THE PERSON TO BLAME, A.K.A. AUTHOR -> Rafael Skellett, 21498314, 2/8/24*
*************************************************************************
P.S. MY '1' KEY DOESN'T WORK, COMMON SUBSTITUTION I USED WAS EITHER 'O' or '0', SORRY FOR CONFUSION

REFERENCES:
Uses yolov5 https://github.com/ultralytics/yolov5