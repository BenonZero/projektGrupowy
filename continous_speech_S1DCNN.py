import os
import tkinter
import customtkinter
import sounddevice as sound
from scipy.io.wavfile import write
from PIL import Image, ImageTk
import threading
import torch
import torchaudio
from torchaudio import transforms
import soundfile
from firstMethodModel import S1DCNN

global trainingDir
global phrase
global phraseType
global searchedFile
global text
global textV
global text1
global textV1
global inputWindow
global searchWindow
global mainMenu

def wordInputKeyboard():
    labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go',
              'happy',
              'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six',
              'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

    inputWordWindow = customtkinter.CTkToplevel()
    inputWordWindow.geometry("1000x100")
    inputWordWindow.title('Fraza do wyszukania')

    global text
    global textV

    text = "Wpisz jedno ze słów: "
    for l in labels:
        text = text + l + ", "

    text = text + "."
    textV = tkinter.StringVar()
    textV.set(text)
    tkinter.Label(inputWordWindow, textvariable=textV).pack(pady=10)

    entry = tkinter.Entry(inputWordWindow)
    entry.pack(padx=30, pady=10)

    def zapiszSlowo():
        global phrase
        phrase = entry.get()

        inputWordWindow.destroy()

    tkinter.Button(inputWordWindow,
              text='Zapisz', command=zapiszSlowo).pack(
                                                        padx=40,
                                                           pady=10)

def clickSearch():
    global phrase
    wordIndex = 78
    labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go',
              'happy',
              'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six',
              'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

    for i in range (35):
        if (phrase == labels[i]):
            wordIndex = i

    model = S1DCNN()

    state = torch.load('model.pt')
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    searching=True

    global searchWindow
    searchWindow = customtkinter.CTkToplevel()
    searchWindow.geometry("600x510")
    searchWindow.title('Szukanie frazy z mikrofonu....')

    def selectClose():
        Searching = False
        inputWindow.destroy()

    global text
    global textV
    global text1
    global textV1

    text = "Szukanie frazy z mikrofonu..."
    textV = tkinter.StringVar()
    textV.set(text)
    tkinter.Label(searchWindow, textvariable=textV).pack(pady=10)

    buttonClose = customtkinter.CTkButton(master=searchWindow, text="Zakończ",
                                          command=selectClose, width=300, height=50)
    buttonClose.pack(pady=10, padx=20)

    plkNum=0
    while(searching==True):
        freq = 44100
        dur = 1  # chyba w sek

        plkNum=plkNum+1
        recording = sound.rec(dur * freq, samplerate=freq, channels=2)
        sound.wait()
        write("recording" + str(plkNum) + ".wav", freq, recording)

        filepath = "recording" + str(plkNum) + ".wav"
        waveform, sample_rate = torchaudio.load(filepath)

        resample_rate = 8000
        resampler = torchaudio.transforms.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
        resampled_waveform = resampler(waveform)
        transform = transforms.MFCC(sample_rate=resample_rate, n_mfcc=13, log_mels=True)
        mfcc = transform(resampled_waveform)

        with torch.no_grad():
            pred = model(mfcc)


        if (pred[0][wordIndex] > 8):                    #Tu zmieniać
            text1 = "Znaleziono podany wyraz!!!"
            textV1 = tkinter.StringVar()
            textV1.set(text1)
            tkinter.Label(searchWindow, textvariable=textV1).pack(pady=10)


    quit()

def clickInfo():
    infoWindow = customtkinter.CTkToplevel()
    infoWindow.geometry("600x250")
    infoWindow.title('Informacje o projekcie')
    text = tkinter.Text(infoWindow)
    text.insert('1.0', 'Wykrywanie słów w mowie ciągłej \n Skład zespołu: \n Jacenty Andruszkiewicz - kierownik \n Michał Kowalewski \n Jakub Kiliańczyk \n Jakub Kwiatkowski \n Piotr Szymański \n Opiekun zespołu: \n dr inż. Maciej Smiatacz \n Katedra Inteligentnych Systemów Interaktywnych \n Link do repozytorium: https://github.com/BenonZero/projektGrupowy')
    text.config(state=tkinter.DISABLED)
    text.pack()



def clickExit():
    quit()

if __name__ == "__main__":
    trainingDir = "brak"
    phrase = "brak"
    phraseType = "brak"
    searchedFile = "brak"

    customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
    customtkinter.set_default_color_theme("dark-blue")  # Themes: blue (default), dark-blue, green

    mainMenu = customtkinter.CTkToplevel()  # create CTk window like you do with the Tk window
    mainMenu.geometry("600x350")
    mainMenu.title('Wykrywanie słów kluczowych w mowie ciągłej')

    imageInputFiles = ImageTk.PhotoImage(Image.open("images/inputFiles.png").resize((50,50), Image.LANCZOS))
    imageNetworkTraining = ImageTk.PhotoImage(Image.open("images/networkTraining.png").resize((50,50), Image.LANCZOS))
    imageSearch = ImageTk.PhotoImage(Image.open("images/search.png").resize((50,50), Image.LANCZOS))
    imageInfo = ImageTk.PhotoImage(Image.open("images/info.png").resize((50,50), Image.LANCZOS))
    imageExit = ImageTk.PhotoImage(Image.open("images/exit.png").resize((50,50), Image.LANCZOS))

    buttonInputFiles = customtkinter.CTkButton(master=mainMenu, text="Podaj słowo szukane", image=imageInputFiles, command=wordInputKeyboard, width=300, height=50, compound="right")
    buttonSearch = customtkinter.CTkButton(master=mainMenu, text="Szukaj frazy w mowie ciągłej ", image=imageSearch, command=threading.Thread(target=clickSearch).start, width=300, height=50, compound="right")
    buttonInfo = customtkinter.CTkButton(master=mainMenu, text="Informacje o projekcie", image=imageInfo, command=clickInfo, width=300, height=50, compound="right")
    buttonExit = customtkinter.CTkButton(master=mainMenu, text="Wyjście", image=imageExit, command=clickExit, width=300, height=50, compound="right")
    buttonInputFiles.pack(pady=10, padx=20)
    buttonSearch.pack(pady=10, padx=20)
    buttonInfo.pack(pady=10, padx=20)
    buttonExit.pack(pady=10, padx=20)

    mainMenu.mainloop()