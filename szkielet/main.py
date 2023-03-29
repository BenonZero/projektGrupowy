import tkinter
import customtkinter
from tkinter import filedialog
from PIL import Image, ImageTk

trainingDir = ""
phrase = ""
searchedFile = ""

def selectTrainingDir():
    trainingDir = filedialog.askdirectory(initialdir="/", title="Select a directory")

def selectPhrase():
    phrase = filedialog.askopenfilename(initialdir="/", title="Select a .wav file", filetypes=(("Audio files", "*.wav*"), ("all files", "*.*")))

def selectSearchedFile():
    searchedFile = filedialog.askopenfilename(initialdir="/", title="Select a .wav file", filetypes=(("Audio files", "*.wav*"), ("all files", "*.*")))

def clickInputFiles():
    inputWindow = customtkinter.CTkToplevel()
    inputWindow.geometry("600x350")
    inputWindow.title('Pliki wejściowe')

    buttonChangeTrainingDir = customtkinter.CTkButton(master=inputWindow, text="Zmień baze plików do treningu (folder)", command=selectTrainingDir, width=300, height=50)
    buttonChangePhrase = customtkinter.CTkButton(master=inputWindow, text="Zmień wyszukiwaną frazę", command=selectPhrase, width=300, height=50)
    buttonChangeSearchedFile = customtkinter.CTkButton(master=inputWindow, text="Zmień plik przeszukiwany", command=selectSearchedFile, width=300, height=50)

    buttonChangeTrainingDir.pack(pady=10, padx=20)
    buttonChangePhrase.pack(pady=10, padx=20)
    buttonChangeSearchedFile.pack(pady=10, padx=20)


def clickNetworkTraining():
    #tu kod trenowania sieci
    quit()

def clickSearch():
    #tu kod wyszukiwania słów kluczowych
    quit()

def clickInfo():
    infoWindow = customtkinter.CTkToplevel()
    infoWindow.geometry("600x350")
    infoWindow.title('Informacje o projekcie')
    text = tkinter.Text(infoWindow)
    text.insert('1.0', 'Wykrywanie słów w mowie ciągłej \n Skład zespołu: \n Jacenty Andruszkiewicz - kierownik \n Michał Kowalewski \n Jakub Kiliańczyk \n Jakub Kwiatkowski \n Piotr Szymański \n Opiekun zespołu: \n dr inż. Maciej Smiatacz \n Katedra Inteligentnych Systemów Interaktywnych')
    text.config(state=tkinter.DISABLED)
    text.pack()


def clickExit():
    quit()


customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("dark-blue")  # Themes: blue (default), dark-blue, green

mainMenu = customtkinter.CTk()  # create CTk window like you do with the Tk window
mainMenu.geometry("600x350")
mainMenu.title('Wykrywanie słów kluczowych w mowie ciągłej')

imageInputFiles = ImageTk.PhotoImage(Image.open("images/inputFiles.png").resize((50,50), Image.LANCZOS))
imageNetworkTraining = ImageTk.PhotoImage(Image.open("images/networkTraining.png").resize((50,50), Image.LANCZOS))
imageSearch = ImageTk.PhotoImage(Image.open("images/search.png").resize((50,50), Image.LANCZOS))
imageInfo = ImageTk.PhotoImage(Image.open("images/info.png").resize((50,50), Image.LANCZOS))
imageExit = ImageTk.PhotoImage(Image.open("images/exit.png").resize((50,50), Image.LANCZOS))

buttonInputFiles = customtkinter.CTkButton(master=mainMenu, text="Zmień baze plików wejściowych", image=imageInputFiles, command=clickInputFiles, width=300, height=50, compound="right")
buttonNetworkTraining = customtkinter.CTkButton(master=mainMenu, text="Trenuj sieci", image=imageNetworkTraining, command=clickNetworkTraining, width=300, height=50, compound="right")
buttonSearch = customtkinter.CTkButton(master=mainMenu, text="Szukaj frazy", image=imageSearch, command=clickSearch, width=300, height=50, compound="right")
buttonInfo = customtkinter.CTkButton(master=mainMenu, text="Informacje o projekcie", image=imageInfo, command=clickInfo, width=300, height=50, compound="right")
buttonExit = customtkinter.CTkButton(master=mainMenu, text="Wyjście", image=imageExit, command=clickExit, width=300, height=50, compound="right")
buttonInputFiles.pack(pady=10, padx=20)
buttonNetworkTraining.pack(pady=10, padx=20)
buttonSearch.pack(pady=10, padx=20)
buttonInfo.pack(pady=10, padx=20)
buttonExit.pack(pady=10, padx=20)

mainMenu.mainloop()