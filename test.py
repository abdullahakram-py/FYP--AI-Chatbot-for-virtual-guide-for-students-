import pyautogui as auto
from time import sleep
import os

def test():
    questions = ["hello", "how are you","how can you help me?", "what is your name?", "what is your age?",
                 "how to get admission in uet taxila","how many departments are there in uet taxila",
                 "how to get my DMC", "how to get my transcript", "how to get English Proficiecy Certificate",
                 "How to get EPC", "Is there any library in uet taxlila",
                "how many credit hours can I take in summer?", "How can I improve my grade in any subject?",
                "What are CLOs?", "What are PLOs?", "lsdjfksdiviojiod", "What the hell are you doing!"]

    for i in range(10):
        sleep(1)
        print("Testing starts in ", 10-i, " seconds")
        

    for index, question in enumerate(questions):
        auto.write(question, interval=0.1)
        auto.press("enter")
        if index == 0:
            sleep(6)
        else:
            sleep(4)
        

if __name__ == '__main__':
    
    while True:
        
        print("=======================")
        print("1. Test Chatbot")
        print("2. Train Again")
        print("3. Exit")
        print("=======================")
        
        try:
            choice = int(input("\n >> "))
        except:
            print("Invalid Input")
            continue
        
        if choice == 1:
            test()
        elif choice == 2:
            os.remove("classes.pkl")
            os.remove("words.pkl")
            os.remove("chatbot_model.h5")
            os.system("python train_chatbot.py")
            print("Training Completed")
        elif choice == 3:
            break
        else:
            print("Invalid Input")
            continue