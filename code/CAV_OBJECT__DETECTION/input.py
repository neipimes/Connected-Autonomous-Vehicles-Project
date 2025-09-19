from pynput import keyboard

class keyboardListener: 
    # create a listener  
    def __init__(self):
        self.lastKey = None
        self.keyboardListener = keyboard.Listener(
        on_press=self.on_key_press)

    def on_key_press(self, key):
        try: 
            self.lastKey = key
            print(f"Pressed: {key.char}")
        except AttributeError: #non-valid chars, eg enter or escape
            self.lastKey = None

    def getLastKey(self):
        if self.lastKey is not None: 
            key = self.lastKey.char
        else: 
            key = self.lastKey
        return key
    
    def initKeyboard(self):
        self.keyboardListener.start()

    def endKeyboard(self):
        self.keyboardListener.stop()
        self.keyboardListener.join()






