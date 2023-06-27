import os
import tkinter
from E2E_KS_Attention_Energy_Scorer.source.vol3.SED3 import SpeechAttDecVol3, SpeechEncVol3
import customtkinter
import sounddevice as sound
from scipy.io.wavfile import write
from PIL import Image, ImageTk
import threading
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms
import soundfile
from S1DCNN import S1DCNN
from tkinter import filedialog
from pydub import AudioSegment

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
global modelif
modelif = "S1DCNN"

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

def clickSearchFromFile():
    global modelif
    global phrase
    wordIndex = 78
    labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go',
              'happy',
              'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six',
              'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

    for i in range(35):
        if (phrase == labels[i]):
            wordIndex = i

    if (modelif == "S1DCNN"):
        model = S1DCNN()
        state = torch.load('model.pt')
        model.load_state_dict(state['model_state_dict'])
        model.eval()

    filename = filedialog.askopenfilename(initialdir="/",
                                        title="Select a File",
                                        filetypes=(("Audio files",
                                                    "*.wav*"),))
    newAudio = AudioSegment.from_wav(filename)

    duration = newAudio.duration_seconds
    found = []


    if modelif == "End2End":

    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_enc = SpeechEncVol3(device, gru_layers=2, hidden_size=256)
        model_dec = SpeechAttDecVol3(device, gru_layers=2, hidden_size=256)

        state2 = torch.load('models\model2\gru2_256_local\GUR4_model_mfcc_v3_e22_l12', map_location=device) # best so far
        model_enc.load_state_dict(state2['model_encoder_state_dict'])
        model_dec.load_state_dict(state2['model_decoder_state_dict'])
        model_enc.eval()
        model_dec.eval()

    for i in range(int(duration)):
        t1 = i * 1000
        t2 = t1 + 1000
        audioPart = newAudio[t1:t2]
        audioPart.export('second' + str(i) + '.wav', format="wav")

        filepath = "second" + str(i) + ".wav"
        waveform, sample_rate = torchaudio.load(filepath)

        if modelif == "End2End":
            transform = transforms.MFCC(sample_rate=sample_rate, n_mfcc=13)
            waveform = transform(waveform)
            padding_value = waveform[0][0][0]
            padding = 81 - waveform.size(2)
            rest = padding % 2
            padding //= 2
            waveform = F.pad(waveform, (padding, padding + rest), value=padding_value)
            waveform = waveform.repeat((1, 1, 1))

            with torch.no_grad():
                emb, h = model_enc(waveform)
                pred = model_dec(emb, h)

            l = pred[0].tolist()
            s = ""
            for c in l:
                s += (chr(c.index(max(c)) + 97))
            padding_val = "{"
            pred = s.replace(padding_val, ' ').strip()

            if pred in labels:
                found.append((i, pred))
        else:
            resample_rate = 8000
            resampler = torchaudio.transforms.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
            resampled_waveform = resampler(waveform)
            transform = transforms.MFCC(sample_rate=resample_rate, n_mfcc=13, log_mels=True)
            mfcc = transform(resampled_waveform)

            with torch.no_grad():
                pred = model(mfcc)
            
            if (pred[0][wordIndex] > 8):  # Tu zmieniać
                found.append(i, labels[wordIndex])

    global searchWindow
    searchWindow = customtkinter.CTkToplevel()
    searchWindow.geometry("600x510")
    searchWindow.title('Szukanie frazy w pliku')

    global text
    global textV
    global text1
    global textV1

    text = "Szukanie frazy z mikrofonu..."
    textV = tkinter.StringVar()
    textV.set(text)
    tkinter.Label(searchWindow, textvariable=textV).pack(pady=10)

    for f in found:
        text1 = f'Znaleziono wyraz "{str(f[1])}" w {str(f[0])} sekundzie pliku.'
        textV1 = tkinter.StringVar()
        textV1.set(text1)
        tkinter.Label(searchWindow, textvariable=textV1).pack(pady=10)


def clickSearch():
    global modelif
    global phrase
    wordIndex = 78
    labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go',
              'happy',
              'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six',
              'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

    for i in range (35):
        if (phrase == labels[i]):
            wordIndex = i
    if modelif == "End2End":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_enc = SpeechEncVol3(device, gru_layers=2, hidden_size=256)
        model_dec = SpeechAttDecVol3(device, gru_layers=2, hidden_size=256)
        state2 = torch.load('models\model2\gru2_256_local\GUR4_model_mfcc_v3_e22_l12', map_location=device) # best so far
        model_enc.load_state_dict(state2['model_encoder_state_dict'])
        model_dec.load_state_dict(state2['model_decoder_state_dict'])
        model_enc.eval()
        model_dec.eval()

    else:
        model = S1DCNN()
        state = torch.load('model.pt')
        model.load_state_dict(state['model_state_dict'])
        model.eval()



    global searching
    searching = True

    global searchWindow
    searchWindow = customtkinter.CTkToplevel()
    searchWindow.geometry("600x510")
    searchWindow.title('Szukanie frazy z mikrofonu....')

    def selectClose():
        global searching
        searching = False
        searchWindow.destroy()

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
        if modelif == "End2End":
            freq2 = 16000
            dur = 1  # chyba w sek

            plkNum=plkNum+1
            recording = sound.rec(int(dur * freq2), samplerate=freq2, channels=2)
            sound.wait()
            write("recording" + str(plkNum) + ".wav", freq2, recording)

            filepath = "recording" + str(plkNum) + ".wav"
            waveform, sample_rate = torchaudio.load(filepath)
            # resample_rate = 8000
            # resampler = torchaudio.transforms.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
            # resampled_waveform = resampler(waveform)
            # transform = transforms.MFCC(sample_rate=resample_rate, n_mfcc=13, log_mels=True)
            # mfcc = transform(resampled_waveform)

            # (1, 16000)
            # (1, 13, 81)

            transform = transforms.MFCC(sample_rate=sample_rate, n_mfcc=13)
            waveform = transform(waveform)
            padding_value = waveform[0][0][0]
            padding = 81 - waveform.size(2)
            rest = padding % 2
            padding //= 2
            waveform = F.pad(waveform, (padding, padding + rest), value=padding_value)
            waveform = waveform.repeat((1, 1, 1))
            with torch.no_grad():
                emb, h = model_enc(waveform)
                pred = model_dec(emb, h)

            l = pred[0].tolist()
            s = ""
            for c in l:
                s += (chr(c.index(max(c)) + 97))
            padding_val = "{"
            pred = s.replace(padding_val, ' ').strip()

            if pred in labels:
                text1 = pred
                textV1 = tkinter.StringVar()
                textV1.set(text1)
                tkinter.Label(searchWindow, textvariable=textV1, font=("Arial", 25)).pack(pady=10)
            
        else:
            freq = 44100
            dur = 1  # chyba w sek

            plkNum=plkNum+1
            recording = sound.rec(int(dur * freq), samplerate=freq, channels=2)
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

def selectModel():
    global modelif

    modelWindow = customtkinter.CTkToplevel()
    modelWindow.geometry("600x550")
    modelWindow.title('Wybór modelu')

    def changeTo1():
        global modelif
        modelif = "S1DCNN"
        modelWindow.destroy()

    def changeTo2():
        global modelif
        modelif = "End2End"
        modelWindow.destroy()

    buttonModel1 = customtkinter.CTkButton(master=modelWindow, text="S1DCNN + wybór słowa szukanego", command=changeTo1, width=300, height=50, compound="right")
    buttonModel2 = customtkinter.CTkButton(master=modelWindow, text="End to end + wykrywanie wszystkich słów", command=changeTo2, width=300, height=50, compound="right")

    buttonModel1.pack()
    buttonModel2.pack()

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
    buttonSearchFromFile = customtkinter.CTkButton(master=mainMenu, text="Szukaj frazy w pliku ", image=imageSearch,
                                           command=threading.Thread(target=clickSearchFromFile).start, width=300, height=50,
                                           compound="right")
    buttonInfo = customtkinter.CTkButton(master=mainMenu, text="Informacje o projekcie", image=imageInfo, command=clickInfo, width=300, height=50, compound="right")
    buttonExit = customtkinter.CTkButton(master=mainMenu, text="Wyjście", image=imageExit, command=clickExit, width=300, height=50, compound="right")
    buttonSelectMethod = customtkinter.CTkButton(master=mainMenu, text="Wybierz model", command=selectModel, width=300, height=50, compound="right")
    
    buttonInputFiles.pack(pady=10, padx=20)
    buttonSearch.pack(pady=10, padx=20)
    buttonSearchFromFile.pack(pady=10, padx=20)
    buttonInfo.pack(pady=10, padx=20)
    buttonSelectMethod.pack(padx=10, pady=20)
    buttonExit.pack(pady=10, padx=20)

    mainMenu.mainloop()